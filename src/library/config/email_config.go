package config

type EmailConfig struct {
	SMTPServer      string   `yaml:"smtpServer"`
	SMTPPort        int      `yaml:"smtpPort"`
	SenderEmail     string   `yaml:"senderEmail"`
	SenderPassword  string   `yaml:"senderPassword"`
	RecipientEmails []string `yaml:"recipientEmails"`
}
