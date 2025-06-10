package email

import (
	"VectorSphere/src/library/config"
	"VectorSphere/src/library/log"
	"crypto/tls"
	"fmt"
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

	var err error
	for i := 0; i <= s.config.MaxRetries; i++ {
		err = s.trySend(addr, s.config.Username, s.config.Password, s.config.From, to, msg)
		if err == nil {
			log.Info("邮件发送成功: To=%v, Subject=%s", to, subject)
			return nil
		}
		log.Warning("邮件发送失败 (尝试 %d/%d): To=%v, Subject=%s, Error: %v", i+1, s.config.MaxRetries+1, to, subject, err)
		if i < s.config.MaxRetries {
			time.Sleep(s.config.RetryInterval)
		}
	}
	return fmt.Errorf("邮件发送失败，已达到最大重试次数: %w", err)
}

func (s *smtpEmailSender) trySend(addr, username, password, from string, to []string, msg []byte) error {
	var client *smtp.Client
	var err error
	connTimeout := s.config.Timeout

	if s.config.UseTLS {
		tlsConfig := &tls.Config{
			ServerName: s.config.Host,
		}
		log.Info("尝试使用TLS连接SMTP服务器: %s", addr)
		conn, dialErr := tls.Dial("tcp", addr, tlsConfig)
		if dialErr == nil {
			client, err = smtp.NewClient(conn, s.config.Host)
			if err != nil {
				log.Error("创建TLS SMTP客户端失败: %v", err)
				// 如果允许不安全连接，并且TLS主要部分失败（如NewClient），则尝试普通连接
				if s.config.AllowInsecure {
					log.Warning("TLS SMTP客户端创建失败，尝试回退到普通SMTP连接")
					return s.sendPlain(addr, username, password, from, to, msg, connTimeout)
				}
				return fmt.Errorf("创建TLS SMTP客户端失败: %w", err)
			}
			log.Info("TLS SMTP客户端创建成功")
		} else {
			log.Warning("TLS连接失败: %v", dialErr)
			if s.config.AllowInsecure {
				log.Warning("TLS连接失败，尝试回退到普通SMTP连接")
				return s.sendPlain(addr, username, password, from, to, msg, connTimeout)
			}
			return fmt.Errorf("TLS连接失败: %w", dialErr)
		}
	} else {
		log.Info("使用普通SMTP连接")
		return s.sendPlain(addr, username, password, from, to, msg, connTimeout)
	}

	defer client.Close()

	if username != "" && password != "" {
		auth := smtp.PlainAuth("", username, password, s.config.Host)
		if err = client.Auth(auth); err != nil {
			log.Error("SMTP认证失败: %v", err)
			return fmt.Errorf("SMTP认证失败: %w", err)
		}
		log.Info("SMTP认证成功")
	}

	if err = client.Mail(from); err != nil {
		log.Error("设置发件人失败 (%s): %v", from, err)
		return fmt.Errorf("设置发件人失败: %w", err)
	}
	log.Info("发件人设置成功: %s", from)

	for _, recipient := range to {
		if err = client.Rcpt(recipient); err != nil {
			log.Error("设置收件人失败 (%s): %v", recipient, err)
			return fmt.Errorf("设置收件人 %s 失败: %w", recipient, err)
		}
		log.Info("收件人设置成功: %s", recipient)
	}

	w, err := client.Data()
	if err != nil {
		log.Error("开启数据传输失败: %v", err)
		return fmt.Errorf("开启数据传输失败: %w", err)
	}
	log.Info("数据传输已开启")

	_, err = w.Write(msg)
	if err != nil {
		log.Error("写入邮件内容失败: %v", err)
		return fmt.Errorf("写入邮件内容失败: %w", err)
	}
	log.Info("邮件内容写入成功")

	err = w.Close()
	if err != nil {
		log.Error("关闭数据传输失败: %v", err)
		return fmt.Errorf("关闭数据传输失败: %w", err)
	}
	log.Info("数据传输关闭成功")

	return client.Quit()
}

func (s *smtpEmailSender) sendPlain(addr, username, password, from string, to []string, msg []byte, timeout time.Duration) error {
	// 注意：net/smtp 的 SendMail 函数内部处理连接和超时，但这里的超时是针对整个操作的
	// 为了更精细的控制，通常会自己管理连接，但 SendMail 提供了便利性
	// 这里我们假设 SendMail 内部有合理的超时，或者依赖外部调用者控制整体超时
	var auth smtp.Auth
	if username != "" && password != "" {
		auth = smtp.PlainAuth("", username, password, s.config.Host)
		log.Info("普通SMTP连接使用认证")
	} else {
		log.Info("普通SMTP连接未使用认证")
	}

	// smtp.SendMail 会自己建立连接、发送然后关闭连接
	// 它不直接接受一个超时参数，但我们可以通过 context 来控制整体操作的超时（如果需要更复杂的场景）
	// 对于简单的重试，我们依赖外部的重试循环
	log.Info("尝试通过 smtp.SendMail 发送邮件 (普通连接) 至 %s", addr)
	err := smtp.SendMail(addr, auth, from, to, msg)
	if err != nil {
		log.Error("smtp.SendMail 发送失败: %v", err)
		return fmt.Errorf("smtp.SendMail 发送失败: %w", err)
	}
	log.Info("通过 smtp.SendMail 发送邮件成功 (普通连接)")
	return nil
}
