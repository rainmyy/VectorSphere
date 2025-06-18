package email

import (
	"VectorSphere/src/library/config"
	"VectorSphere/src/library/logger"
	"crypto/tls"
	"fmt"
	"net"
	"net/smtp"
	"strings"
	"time"
)

// EmailSender 定义邮件发送器接口
type EmailSender interface {
	SendEmail(to []string, subject string, body string) error
}

// smtpEmailSender 是 EmailSender 的SMTP实现
type smtpEmailSender struct {
	config config.EmailConfig
}

// NewSMTPEmailSender 创建一个新的 SMTP邮件发送器实例
func NewSMTPEmailSender(config config.EmailConfig) EmailSender {
	if config.Timeout <= 0 {
		config.Timeout = 30 * time.Second // 默认超时时间
	}
	if config.MaxRetries < 0 {
		config.MaxRetries = 0 // 默认不重试
	}
	if config.RetryInterval <= 0 {
		config.RetryInterval = 5 * time.Second // 默认重试间隔
	}
	return &smtpEmailSender{config: config}
}

// SendEmail 实现发送邮件逻辑
func (s *smtpEmailSender) SendEmail(to []string, subject string, body string) error {
	if len(to) == 0 {
		return fmt.Errorf("收件人列表不能为空")
	}
	if s.config.From == "" {
		return fmt.Errorf("发件人地址不能为空")
	}
	if s.config.Host == "" {
		return fmt.Errorf("SMTP服务器地址不能为空")
	}
	if s.config.Port == 0 {
		return fmt.Errorf("SMTP服务器端口不能为空")
	}

	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	msg := []byte("To: " + strings.Join(to, ",") + "\r\n" +
		"From: " + s.config.From + "\r\n" +
		"Subject: " + subject + "\r\n" +
		"Content-Type: text/html; charset=UTF-8\r\n" +
		"\r\n" +
		body)

	var lastErr error
	for i := 0; i <= s.config.MaxRetries; i++ {
		err := s.trySendOnce(addr, s.config.Username, s.config.Password, s.config.From, to, msg)
		if err == nil {
			logger.Info("邮件发送成功: To=%v, Subject=%s", to, subject)
			return nil
		}
		lastErr = err
		logger.Warning("邮件发送尝试失败 (%d/%d): To=%v, Subject=%s, Error: %v", i+1, s.config.MaxRetries+1, to, subject, err)
		if i < s.config.MaxRetries {
			logger.Info("等待 %v 后重试...", s.config.RetryInterval)
			time.Sleep(s.config.RetryInterval)
		}
	}
	return fmt.Errorf("邮件发送失败，已达到最大重试次数 (%d): %w", s.config.MaxRetries+1, lastErr)
}

// trySendOnce 尝试一次性发送邮件，处理TLS、回退到普通连接（如果允许）或直接普通连接。
func (s *smtpEmailSender) trySendOnce(addr, username, password, from string, to []string, msg []byte) error {
	if s.config.UseTLS {
		logger.Info("尝试通过TLS发送邮件至 %s", addr)
		err := s.sendOverTLS(addr, username, password, from, to, msg)
		if err != nil {
			logger.Warning("通过TLS发送邮件失败: %v", err)
			if s.config.AllowInsecure {
				logger.Info("由于AllowInsecure为true，回退到普通SMTP连接。")
				return s.sendOverPlain(addr, username, password, from, to, msg)
			}
			return fmt.Errorf("TLS发送失败且不允许回退到不安全连接: %w", err)
		}
		logger.Info("邮件通过TLS成功发送。")
		return nil
	}

	logger.Info("尝试通过普通SMTP发送邮件至 %s", addr)
	err := s.sendOverPlain(addr, username, password, from, to, msg)
	if err != nil {
		return fmt.Errorf("普通SMTP发送失败: %w", err)
	}
	logger.Info("邮件通过普通SMTP成功发送。")
	return nil
}

// sendOverTLS 处理通过TLS连接发送邮件的逻辑。
func (s *smtpEmailSender) sendOverTLS(addr, username, password, from string, to []string, msg []byte) error {
	tlsConfig := &tls.Config{
		ServerName: s.config.Host,
		MinVersion: tls.VersionTLS12, // 考虑为安全设置最小TLS版本
	}

	conn, err := tls.DialWithDialer(&net.Dialer{Timeout: s.config.Timeout}, "tcp", addr, tlsConfig)
	if err != nil {
		return fmt.Errorf("TLS拨号失败 %s: %w", addr, err)
	}
	defer conn.Close()

	client, err := smtp.NewClient(conn, s.config.Host)
	if err != nil {
		return fmt.Errorf("TLS拨号后创建SMTP客户端失败 %s: %w", s.config.Host, err)
	}
	defer client.Close()

	if username != "" && password != "" {
		auth := smtp.PlainAuth("", username, password, s.config.Host)
		if err = client.Auth(auth); err != nil {
			return fmt.Errorf("SMTP认证失败 %s: %w", s.config.Host, err)
		}
		logger.Info("通过TLS成功进行SMTP认证。")
	}

	if err = client.Mail(from); err != nil {
		return fmt.Errorf("SMTP MAIL FROM命令失败 %s: %w", from, err)
	}
	for _, recipient := range to {
		if err = client.Rcpt(recipient); err != nil {
			return fmt.Errorf("SMTP RCPT TO命令失败 %s: %w", recipient, err)
		}
	}

	w, err := client.Data()
	if err != nil {
		return fmt.Errorf("SMTP DATA命令失败: %w", err)
	}
	defer w.Close() // 确保在写入器不再需要时关闭

	if _, err = w.Write(msg); err != nil {
		return fmt.Errorf("写入邮件消息失败: %w", err)
	}
	// w.Close() 将在 defer 中调用，或者在这里显式调用然后从defer中移除
	// 对于 DATA 命令，必须在所有数据写入后调用 Close() 来结束 DATA 序列
	if err = w.Close(); err != nil { // 显式关闭以捕获错误
		return fmt.Errorf("关闭邮件数据写入器失败: %w", err)
	}

	return nil // 成功，client.Quit() 将由 defer client.Close() 处理
}

// sendOverPlain 处理通过普通SMTP连接发送邮件的逻辑（使用smtp.SendMail）。
func (s *smtpEmailSender) sendOverPlain(addr, username, password, from string, to []string, msg []byte) error {
	var auth smtp.Auth
	if username != "" && password != "" {
		auth = smtp.PlainAuth("", username, password, s.config.Host)
		logger.Info("为 %s 使用带认证的普通SMTP。", s.config.Host)
	} else {
		logger.Info("为 %s 使用不带认证的普通SMTP。", s.config.Host)
	}

	err := smtp.SendMail(addr, auth, from, to, msg)
	if err != nil {
		return fmt.Errorf("smtp.SendMail失败 %s: %w", addr, err)
	}
	return nil
}
