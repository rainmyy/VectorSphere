package search

import (
	"fmt"
	"strconv"
	"strings"
	"text/scanner"
	"unicode"
)

/**
- 重写了 nextToken 以手动处理字符流，而不是完全依赖 text/scanner 的 Scan() 。这能更好地控制多字符操作符（如 >= , <= , != ）和错误处理。
- 添加了 skipWhitespace , readIdentifier , readNumber , readString 等辅助函数。
- 改进了对字符串字面量的处理（简单版本，未处理转义）。
- AST节点 :
- 为 SelectStatement 添加了 Fields []Expression 以支持选择特定字段（当前解析逻辑仍简化）。
- 添加了 StarExpression 代表 SELECT * 。
- LiteralExpression 的 Value 改为 interface{} 以存储实际Go类型。
- 为AST节点添加了 String() 方法，方便调试和打印AST。
- Parser :
- ParseStatement 现在调用 parseSelectStatement 。
- parseSelectStatement 开始填充 Fields , From , Where , OrderBy , Limit , Offset 。
- parseExpression 是Pratt解析器的核心，用于处理表达式和操作符优先级。当前它非常基础。
- 添加了 parseIdentifier , parseNumberLiteral , parseStringLiteral , parseStarExpression , parseKeywordLiteral (for true/false/null), parseGroupedExpression , parseInfixExpression 。
- 添加了 parseOrderByClause , parseLimitClause , parseOffsetClause 的基本实现。
- 错误处理通过 p.errorf 和 p.peekError 收集到 p.errors 。
- 错误处理 :
- Lexer现在可以生成 TokenError 。
- Parser收集Lexer错误和自身的解析错误。
- ParseQueryWithAST 返回错误列表。
- AST到查询参数 ( ConvertASTToParsedQuery ) :
- 新增 ParsedQuery 结构体（可以根据需要调整，使其更接近您系统所需参数）。
- 新增 FilterCondition 结构体。
- ConvertASTToParsedQuery 函数遍历 SelectStatement AST，并将信息提取到 ParsedQuery 结构中。
- extractConditions 是一个非常简化的函数，用于从 WHERE 子句的AST中提取条件。一个完整的实现需要正确处理 AND , OR , NOT 以及括号构成的复杂逻辑树。
- Lexer ( Lexer , NewLexer , readChar , nextToken , skipWhitespaceAndComments , readString , readNumber ) :
- Lexer 结构体增加了 line 和 column 用于更精确的错误定位。
- readChar 现在会更新行号和列号。
- skipWhitespaceAndComments 函数被添加用于跳过空格、制表符、换行符以及 SQL 风格的单行注释和多行注释readString 进行了改进，以初步支持 SQL 中的 '' (两个单引号表示一个单引号字符) 和常见的C风格转义序列 (如 \n , \t , \' , \\ )。更复杂的转义规则可能后续还需要完善。
- readNumber 稍微调整以更好地区分整数和浮点数（尽管这里的逻辑还可以进一步加强以符合SQL数字字面量的完整规范）。
- Parser ( NewParser , parseSelectStatement , parseSelectList , expectKeyword , expect ) :
- parseSelectStatement : 调整了对 FROM 和表名的期望，使用新的 expectKeyword 和 expect 辅助函数，这些函数在检查失败时会记录错误并尝试继续（或者根据策略停止）。
- parseSelectList : 这是新增的核心逻辑，用于解析 SELECT 后的字段列表。它可以处理 SELECT * , SELECT field1 , SELECT field1, field2, ... 。它会循环查找逗号并解析后续的表达式。当前的表达式解析还比较简单，主要处理标识符和星号。
- expectKeyword 和 expect : 新的辅助函数，用于简化对当前 Token 的检查和消费，并在不匹配时记录错误。
- precedences 和 NewParser 中的 registerInfix 为后续处理 AND / OR 等逻辑操作符做了准备，将 TokenKeyword 也注册为潜在的中缀操作符类型。
*/

// TokenType 定义了词法单元的类型
type TokenType int

const (
	TokenEOF TokenType = iota
	TokenError
	TokenIdent       // 标识符 (e.g., table_name, field_name)
	TokenKeyword     // SQL 关键字 (SELECT, FROM, WHERE, etc.)
	TokenString      // 字符串字面量 'abc'
	TokenNumber      // 数字字面量 123
	TokenOperator    // 操作符 (=, AND, OR, etc.)
	TokenLParen      // (
	TokenRParen      // )
	TokenComma       // ,
	TokenSemicolon   // ;
	TokenPlaceholder // ? for prepared statements (future use)
	TokenStar        // * (for SELECT *)
)

var tokenKeywords = map[string]TokenType{
	"select": TokenKeyword,
	"from":   TokenKeyword,
	"where":  TokenKeyword,
	"and":    TokenKeyword,
	"or":     TokenKeyword,
	"limit":  TokenKeyword,
	"offset": TokenKeyword,
	"order":  TokenKeyword,
	"by":     TokenKeyword,
	"asc":    TokenKeyword,
	"desc":   TokenKeyword,
	"true":   TokenKeyword, // Boolean literals as keywords for simplicity
	"false":  TokenKeyword,
	"null":   TokenKeyword,
}

// GroupedExpression for ( expression )
type GroupedExpression struct {
	Token         Token // The '(' token
	SubExpression Expression
}

func (ge *GroupedExpression) expressionNode()      {}
func (ge *GroupedExpression) TokenLiteral() string { return ge.Token.Literal }
func (ge *GroupedExpression) String() string {
	var out strings.Builder
	out.WriteString("(")
	if ge.SubExpression != nil {
		out.WriteString(ge.SubExpression.String())
	}
	out.WriteString(")")
	return out.String()
}

// Token 结构体表示一个词法单元
type Token struct {
	Type    TokenType
	Literal string
	Pos     scanner.Position
}

// Lexer 结构体
type Lexer struct {
	s       scanner.Scanner
	input   string
	pos     int  // current position in input (for manual multi-char op handling)
	readPos int  // current reading position in input (after current char)
	ch      byte // current char under examination
	line    int  // current line number, for better error reporting
	column  int  // current column number, for better error reporting
}

// NewLexer 创建一个新的词法分析器
func NewLexer(input string) *Lexer {
	l := &Lexer{input: input}
	l.s.Init(strings.NewReader(input))
	l.s.Mode = scanner.ScanIdents | scanner.ScanFloats | scanner.ScanStrings | scanner.ScanComments | scanner.ScanRawStrings
	// Customize IsIdentRune to allow hyphens in idents (e.g., for table names like my-table)
	l.s.IsIdentRune = func(ch rune, i int) bool {
		return unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_' || ch == '-'
	}
	l.readChar() // Initialize l.ch and l.pos, l.readPos
	return l
}

func (l *Lexer) readChar() {
	if l.readPos >= len(l.input) {
		l.ch = 0 // EOF
	} else {
		l.ch = l.input[l.readPos]
	}
	l.pos = l.readPos
	l.readPos++

	if l.ch == '\n' {
		l.line++
		l.column = 0
	} else {
		l.column++
	}
}

func (l *Lexer) peekChar() byte {
	if l.readPos >= len(l.input) {
		return 0
	}
	return l.input[l.readPos]
}
func (l *Lexer) currentCharPos() scanner.Position {
	// Approximate position, as we are manually tracking line and column
	// For more precise offset, we might need to sum bytes read.
	return scanner.Position{Line: l.line, Column: l.column, Offset: l.pos}
}

// nextToken 从输入中读取下一个Token (改进版)
func (l *Lexer) nextToken() Token {
	var tok Token

	l.skipWhitespaceAndComments()

	currentPos := l.currentCharPos()

	switch l.ch {
	case '=':
		tok = Token{Type: TokenOperator, Literal: string(l.ch), Pos: currentPos}
	case '>':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: TokenOperator, Literal: string(ch) + string(l.ch), Pos: currentPos}
		} else {
			tok = Token{Type: TokenOperator, Literal: string(l.ch), Pos: currentPos}
		}
	case '<':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: TokenOperator, Literal: string(ch) + string(l.ch), Pos: currentPos}
		} else if l.peekChar() == '>' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: TokenOperator, Literal: string(ch) + string(l.ch), Pos: currentPos} // <> for !=
		} else {
			tok = Token{Type: TokenOperator, Literal: string(l.ch), Pos: currentPos}
		}
	case '!':
		if l.peekChar() == '=' {
			ch := l.ch
			l.readChar()
			tok = Token{Type: TokenOperator, Literal: string(ch) + string(l.ch), Pos: currentPos}
		} else {
			tok = Token{Type: TokenError, Literal: "unexpected token: " + string(l.ch), Pos: currentPos}
		}
	case '(':
		tok = Token{Type: TokenLParen, Literal: string(l.ch), Pos: currentPos}
	case ')':
		tok = Token{Type: TokenRParen, Literal: string(l.ch), Pos: currentPos}
	case ',':
		tok = Token{Type: TokenComma, Literal: string(l.ch), Pos: currentPos}
	case ';':
		tok = Token{Type: TokenSemicolon, Literal: string(l.ch), Pos: currentPos}
	case '*':
		tok = Token{Type: TokenStar, Literal: string(l.ch), Pos: currentPos}
	case '\'': // Start of a string literal
		tok.Pos = currentPos
		tok.Type = TokenString
		tok.Literal = l.readString()
	case 0: // EOF
		tok = Token{Type: TokenEOF, Literal: "EOF", Pos: currentPos}
	default:
		if isLetter(l.ch) {
			tok.Pos = currentPos
			tok.Literal = l.readIdentifier()
			lowerLit := strings.ToLower(tok.Literal)
			if keywordType, isKeyword := tokenKeywords[lowerLit]; isKeyword {
				tok.Type = keywordType
			} else {
				tok.Type = TokenIdent
			}
			return tok // readIdentifier already advanced, so return early
		} else if unicode.IsDigit(rune(l.ch)) {
			tok.Pos = currentPos
			tok.Type = TokenNumber
			tok.Literal = l.readNumber()
			return tok // readNumber already advanced, so return early
		} else {
			tok = Token{Type: TokenError, Literal: fmt.Sprintf("unexpected character: %c at %s", l.ch, currentPos.String()), Pos: currentPos}
		}
	}
	l.readChar()
	return tok
}

func (l *Lexer) skipWhitespaceAndComments() {
	for {
		if l.ch == ' ' || l.ch == '\t' || l.ch == '\n' || l.ch == '\r' {
			l.readChar()
			continue
		}
		// Single line comment --
		if l.ch == '-' && l.peekChar() == '-' {
			for l.ch != '\n' && l.ch != 0 {
				l.readChar()
			}
			continue
		}
		// Multi-line comment /* ... */
		if l.ch == '/' && l.peekChar() == '*' {
			l.readChar() // consume /
			l.readChar() // consume *
			for !(l.ch == '*' && l.peekChar() == '/') && l.ch != 0 {
				l.readChar()
			}
			if l.ch != 0 { // if not EOF
				l.readChar() // consume *
				l.readChar() // consume /
			}
			continue
		}
		break
	}
}

func (l *Lexer) skipWhitespace() {
	for l.ch == ' ' || l.ch == '\t' || l.ch == '\n' || l.ch == '\r' {
		l.readChar()
	}
}

func (l *Lexer) readIdentifier() string {
	position := l.pos
	for isLetter(l.ch) || unicode.IsDigit(rune(l.ch)) || l.ch == '-' { // allow digits and hyphens in idents
		l.readChar()
	}
	return l.input[position:l.pos]
}

func (l *Lexer) readNumber() string {
	position := l.pos
	isFloat := false
	for unicode.IsDigit(rune(l.ch)) || (l.ch == '.' && !isFloat && unicode.IsDigit(rune(l.peekChar()))) {
		if l.ch == '.' {
			isFloat = true
		}
		l.readChar()
	}
	return l.input[position:l.pos]
}

func (l *Lexer) readString() string {
	// Skip the opening quote
	var sb strings.Builder
	for {
		l.readChar()
		if l.ch == '\'' {
			// Handle escaped quote: '' is a single quote in SQL strings
			if l.peekChar() == '\'' {
				sb.WriteByte('\'')
				l.readChar() // consume the second quote
				continue
			}
			break // End of string
		} else if l.ch == '\\' { // Handle standard backslash escapes
			l.readChar() // consume backslash
			switch l.ch {
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case 'r':
				sb.WriteByte('\r')
			case '\'':
				sb.WriteByte('\'')
			case '\\':
				sb.WriteByte('\\')
			default:
				// Or record an error for unsupported escape sequence
				sb.WriteByte('\\') // Store as is, or handle error
				sb.WriteByte(l.ch)
			}
		} else if l.ch == 0 { // EOF before closing quote
			// This is an error condition, could be handled by returning TokenError from nextToken
			break
		} else {
			sb.WriteByte(l.ch)
		}
	}
	// l.readChar() // The main loop in nextToken will consume the closing quote if loop broke on '
	return sb.String()
}

func isLetter(ch byte) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_'
}

type ASTNode interface {
	TokenLiteral() string
	String() string
}

type Statement interface {
	ASTNode
	statementNode()
}

type Expression interface {
	ASTNode
	expressionNode()
}

// SelectStatement AST 节点
type SelectStatement struct {
	Token   Token        // The 'SELECT' token
	Fields  []Expression // e.g., *, field1, COUNT(*)
	From    *TableIdentifier
	Where   Expression // Can be nil
	OrderBy *OrderByClause
	Limit   *LimitClause
	Offset  *OffsetClause
}

func (ss *SelectStatement) statementNode()       {}
func (ss *SelectStatement) TokenLiteral() string { return ss.Token.Literal }

func (ss *SelectStatement) String() string {
	var out strings.Builder
	out.WriteString("SELECT ")
	fieldsStr := []string{}
	for _, f := range ss.Fields {
		fieldsStr = append(fieldsStr, f.String())
	}
	out.WriteString(strings.Join(fieldsStr, ", "))
	out.WriteString(" FROM ")
	out.WriteString(ss.From.Value)
	if ss.Where != nil {
		out.WriteString(" WHERE ")
		out.WriteString(ss.Where.String())
	}
	if ss.OrderBy != nil {
		out.WriteString(" ")
		out.WriteString(ss.OrderBy.String())
	}
	if ss.Limit != nil {
		out.WriteString(" ")
		out.WriteString(ss.Limit.String())
	}
	if ss.Offset != nil {
		out.WriteString(" ")
		out.WriteString(ss.Offset.String())
	}
	return out.String()
}

// Identifier AST Node (can be a field or table name)
type Identifier struct {
	Token Token // The identifier token
	Value string
}

func (i *Identifier) expressionNode()      {}
func (i *Identifier) TokenLiteral() string { return i.Token.Literal }
func (i *Identifier) String() string       { return i.Value }

// TableIdentifier is now just an Identifier, or could be more complex if schemas are involved
type TableIdentifier Identifier

// StarExpression for SELECT *
type StarExpression struct {
	Token Token // The '*' token
}

func (se *StarExpression) expressionNode()      {}
func (se *StarExpression) TokenLiteral() string { return se.Token.Literal }
func (se *StarExpression) String() string       { return "*" }

// LiteralExpression (e.g., string, number, boolean)
type LiteralExpression struct {
	Token Token
	Value interface{} // Store the actual Go type (string, int64, float64, bool)
}

func (le *LiteralExpression) expressionNode()      {}
func (le *LiteralExpression) TokenLiteral() string { return le.Token.Literal }
func (le *LiteralExpression) String() string       { return le.Token.Literal } // Or fmt.Sprintf for actual value

// BinaryExpression (e.g., field = 'value', score > 0.5)
type BinaryExpression struct {
	Token    Token // The operator token, e.g., '='
	Left     Expression
	Operator string
	Right    Expression
}

func (be *BinaryExpression) expressionNode()      {}
func (be *BinaryExpression) TokenLiteral() string { return be.Token.Literal }
func (be *BinaryExpression) String() string {
	var out strings.Builder
	out.WriteString("(")
	out.WriteString(be.Left.String())
	out.WriteString(" " + be.Operator + " ")
	out.WriteString(be.Right.String())
	out.WriteString(")")
	return out.String()
}

// OrderByClause AST 节点
type OrderByClause struct {
	Token     Token // The 'ORDER' token
	Field     Expression
	Direction string // "ASC" or "DESC"
}

func (obc *OrderByClause) TokenLiteral() string { return obc.Token.Literal }
func (obc *OrderByClause) String() string {
	return fmt.Sprintf("ORDER BY %s %s", obc.Field.String(), obc.Direction)
}

// LimitClause AST 节点
type LimitClause struct {
	Token Token // The 'LIMIT' token
	Value int64
}

func (lc *LimitClause) TokenLiteral() string { return lc.Token.Literal }
func (lc *LimitClause) String() string       { return fmt.Sprintf("LIMIT %d", lc.Value) }

// OffsetClause AST 节点
type OffsetClause struct {
	Token Token // The 'OFFSET' token
	Value int64
}

func (oc *OffsetClause) TokenLiteral() string { return oc.Token.Literal }
func (oc *OffsetClause) String() string       { return fmt.Sprintf("OFFSET %d", oc.Value) }

// Operator precedence
const (
	_ int = iota
	LOWEST
	LOGICALOR   // OR
	LOGICALAND  // AND
	EQUALS      // ==, !=
	LESSGREATER // > or <
	SUM         // +
	PRODUCT     // *
	PREFIX      // -X or !X
	CALL        // myFunction(X)
)

var precedences = map[TokenType]int{
	TokenOperator: EQUALS, // Default for comparison ops like =, !=, <, >
	// Specific keywords for logical operators will need their own precedence
	// We'll handle AND/OR based on their TokenKeyword type in parseInfixExpression
}

// getPrecedence returns the precedence of the current token.
// It handles both operator tokens and keyword tokens that act as operators (AND, OR).
func (p *Parser) getPrecedence(tok Token) int {
	if typ, ok := precedences[tok.Type]; ok && tok.Type == TokenOperator {
		return typ
	}
	if tok.Type == TokenKeyword {
		switch strings.ToLower(tok.Literal) {
		case "and":
			return LOGICALAND
		case "or":
			return LOGICALOR
		}
	}
	return LOWEST
}

type (
	prefixParseFn func() Expression
	infixParseFn  func(Expression) Expression
)

type Parser struct {
	lexer *Lexer

	curToken  Token
	peekToken Token

	errors []string

	prefixParseFns map[TokenType]prefixParseFn
	infixParseFns  map[TokenType]infixParseFn
}

func NewParser(l *Lexer) *Parser {
	p := &Parser{
		lexer:          l,
		errors:         []string{},
		prefixParseFns: make(map[TokenType]prefixParseFn),
		infixParseFns:  make(map[TokenType]infixParseFn),
	}

	p.registerPrefix(TokenIdent, p.parseIdentifier)
	p.registerPrefix(TokenNumber, p.parseNumberLiteral)
	p.registerPrefix(TokenString, p.parseStringLiteral)
	p.registerPrefix(TokenStar, p.parseStarExpression)
	p.registerPrefix(TokenKeyword, p.parseKeywordLiteral)   // For true, false, null
	p.registerPrefix(TokenLParen, p.parseGroupedExpression) // Register for '('

	p.registerInfix(TokenOperator, p.parseInfixExpression) // For =, >, <, etc.
	p.registerInfix(TokenKeyword, p.parseInfixExpression)  // For AND, OR

	p.nextToken() // Set curToken
	p.nextToken() // Set peekToken
	return p
}

func (p *Parser) Errors() []string {
	return p.errors
}

func (p *Parser) nextToken() {
	p.curToken = p.peekToken
	p.peekToken = p.lexer.nextToken()
	// Skip over error tokens from lexer, but record them
	for p.peekToken.Type == TokenError {
		p.errors = append(p.errors, fmt.Sprintf("Lexer error at %s: %s", p.peekToken.Pos.String(), p.peekToken.Literal))
		p.curToken = p.peekToken // Consume the error token
		p.peekToken = p.lexer.nextToken()
	}
}

func (p *Parser) registerPrefix(tokenType TokenType, fn prefixParseFn) {
	p.prefixParseFns[tokenType] = fn
}

func (p *Parser) registerInfix(tokenType TokenType, fn infixParseFn) {
	p.infixParseFns[tokenType] = fn
}

func (p *Parser) ParseStatement() Statement {
	switch p.curToken.Type {
	case TokenKeyword:
		if strings.ToLower(p.curToken.Literal) == "select" {
			return p.parseSelectStatement()
		}
	}
	p.errorf(p.curToken.Pos, "no parsing function for statement starting with %s (%s)", p.curToken.Literal, p.curToken.Type)
	return nil
}

func (p *Parser) parseSelectStatement() *SelectStatement {
	stmt := &SelectStatement{Token: p.curToken}

	p.nextToken() // Consume SELECT

	stmt.Fields = p.parseSelectList()

	if !p.expectKeyword("from") { // Changed from expectPeekKeyword
		return nil
	}

	if !p.expect(TokenIdent) { // Changed from expectPeek
		return nil
	}
	stmt.From = &TableIdentifier{Token: p.curToken, Value: p.curToken.Literal}
	p.nextToken() // Consume table name

	if p.curTokenIsKeyword("where") {
		p.nextToken() // Consume WHERE
		stmt.Where = p.parseExpression(LOWEST)
	}

	// ORDER BY, LIMIT, OFFSET parsing (remains similar for now)
	if p.curTokenIsKeyword("order") {
		stmt.OrderBy = p.parseOrderByClause()
	}

	if p.curTokenIsKeyword("limit") {
		stmt.Limit = p.parseLimitClause()
	}

	if p.curTokenIsKeyword("offset") {
		stmt.Offset = p.parseOffsetClause()
	}

	return stmt
}

// Helper to expect a specific keyword (consumes it if found)
func (p *Parser) expectKeyword(keyword string) bool {
	if p.curTokenIsKeyword(keyword) {
		p.nextToken()
		return true
	}
	p.errorf(p.curToken.Pos, "expected keyword '%s', got %s (%s) instead", keyword, p.curToken.Literal, p.curToken.Type)
	return false
}

// Helper to expect a specific token type (consumes it if found)
func (p *Parser) expect(tokenType TokenType) bool {
	if p.curTokenIs(tokenType) {
		p.nextToken()
		return true
	}
	p.errorf(p.curToken.Pos, "expected token type %v, got %s (%s) instead", tokenType, p.curToken.Literal, p.curToken.Type)
	return false
}

func (p *Parser) curTokenIs(t TokenType) bool {
	return p.curToken.Type == t
}

func (p *Parser) peekTokenIs(t TokenType) bool {
	return p.peekToken.Type == t
}

func (p *Parser) curTokenIsKeyword(keyword string) bool {
	return p.curToken.Type == TokenKeyword && strings.ToLower(p.curToken.Literal) == keyword
}

func (p *Parser) peekTokenIsKeyword(keyword string) bool {
	return p.peekToken.Type == TokenKeyword && strings.ToLower(p.peekToken.Literal) == keyword
}

func (p *Parser) parseSelectList() []Expression {
	list := []Expression{}

	if p.curTokenIs(TokenStar) {
		list = append(list, p.parseStarExpression())
		// Check for more fields after *, e.g., SELECT *, count(id) - this is more complex
		// For now, if we see a star, we assume it's the only item or the first of a few.
		if p.peekTokenIs(TokenComma) {
			p.nextToken() // Consume '*'
			p.nextToken() // Consume ','
		} else {
			p.nextToken() // Consume '*'
			return list   // If only '*', return
		}
	}

	// Parse first expression
	expr := p.parseExpression(LOWEST)
	if expr != nil {
		list = append(list, expr)
	}

	for p.peekTokenIs(TokenComma) {
		p.nextToken() // Consume previous expression's last token or ','
		p.nextToken() // Consume ',' or the token starting the new expression
		expr = p.parseExpression(LOWEST)
		if expr != nil {
			list = append(list, expr)
		} else {
			// Error: expected expression after comma
			p.errorf(p.curToken.Pos, "expected expression after comma in select list")
			return nil // Or try to recover
		}
	}

	return list
}

func (p *Parser) parseExpression(precedence int) Expression {
	prefix := p.prefixParseFns[p.curToken.Type]
	if prefix == nil {
		p.errorf(p.curToken.Pos, "no prefix parse function for %s (%s)", p.curToken.Literal, p.curToken.Type)
		return nil
	}
	leftExp := prefix()

	for !p.peekTokenIs(TokenSemicolon) && precedence < p.getPrecedence(p.peekToken) {
		infix := p.infixParseFns[p.peekToken.Type]
		if infix == nil {
			return leftExp // No infix operator found, or it's of lower precedence
		}
		p.nextToken() // Consume the operator
		leftExp = infix(leftExp)
	}

	return leftExp
}

func (p *Parser) isEndOfClause(tok Token) bool {
	// Helper to stop expression parsing before next clause keyword
	if tok.Type == TokenKeyword {
		switch strings.ToLower(tok.Literal) {
		case "from", "where", "order", "limit", "offset":
			return true
		}
	}
	return tok.Type == TokenEOF || tok.Type == TokenSemicolon || tok.Type == TokenRParen
}

func (p *Parser) parseIdentifier() Expression {
	return &Identifier{Token: p.curToken, Value: p.curToken.Literal}
}

func (p *Parser) parseStarExpression() Expression {
	return &StarExpression{Token: p.curToken}
}

func (p *Parser) parseNumberLiteral() Expression {
	lit := &LiteralExpression{Token: p.curToken}
	// Try parsing as int, then float
	val, err := strconv.ParseInt(p.curToken.Literal, 0, 64)
	if err == nil {
		lit.Value = val
		return lit
	}
	valFloat, err := strconv.ParseFloat(p.curToken.Literal, 64)
	if err == nil {
		lit.Value = valFloat
		return lit
	}
	p.errorf(p.curToken.Pos, "could not parse %q as number", p.curToken.Literal)
	return nil
}

func (p *Parser) parseStringLiteral() Expression {
	return &LiteralExpression{Token: p.curToken, Value: p.curToken.Literal}
}

func (p *Parser) parseKeywordLiteral() Expression {
	switch strings.ToLower(p.curToken.Literal) {
	case "true":
		return &LiteralExpression{Token: p.curToken, Value: true}
	case "false":
		return &LiteralExpression{Token: p.curToken, Value: false}
	case "null": // Represent SQL NULL, perhaps with a specific Go type or nil
		return &LiteralExpression{Token: p.curToken, Value: nil}
	}
	p.errorf(p.curToken.Pos, "unexpected keyword literal %s", p.curToken.Literal)
	return nil
}

func (p *Parser) parseGroupedExpression() Expression {
	startToken := p.curToken // Keep the '(' token for the AST node
	p.nextToken()            // Consume '('

	subExpression := p.parseExpression(LOWEST)

	if !p.expect(TokenRParen) { // Expect and consume ')'
		return nil
	}

	return &GroupedExpression{
		Token:         startToken,
		SubExpression: subExpression,
	}
}

func (p *Parser) parseInfixExpression(left Expression) Expression {
	expression := &BinaryExpression{
		Token:    p.curToken,
		Operator: p.curToken.Literal, // Could be '=', 'AND', 'OR', etc.
		Left:     left,
	}

	precedence := p.getPrecedence(p.curToken)
	p.nextToken() // Consume the operator token
	expression.Right = p.parseExpression(precedence)

	return expression
}

func (p *Parser) parseOrderByClause() *OrderByClause {
	clause := &OrderByClause{Token: p.curToken} // ORDER token
	if !p.expectPeekKeyword("by") {
		return nil
	}
	p.nextToken()                      // Consume BY
	p.nextToken()                      // Consume field identifier
	clause.Field = p.parseIdentifier() // Simplified could be more complex expression

	clause.Direction = "ASC" // Default
	if p.peekTokenIsKeyword("asc") || p.peekTokenIsKeyword("desc") {
		p.nextToken()
		clause.Direction = strings.ToUpper(p.curToken.Literal)
	}
	return clause
}

func (p *Parser) parseLimitClause() *LimitClause {
	clause := &LimitClause{Token: p.curToken} // LIMIT token
	if !p.expectPeek(TokenNumber) {
		return nil
	}
	val, err := strconv.ParseInt(p.curToken.Literal, 10, 64)
	if err != nil {
		p.errorf(p.curToken.Pos, "could not parse LIMIT value %s: %v", p.curToken.Literal, err)
		return nil
	}
	clause.Value = val
	return clause
}

func (p *Parser) parseOffsetClause() *OffsetClause {
	clause := &OffsetClause{Token: p.curToken} // OFFSET token
	if !p.expectPeek(TokenNumber) {
		return nil
	}
	val, err := strconv.ParseInt(p.curToken.Literal, 10, 64)
	if err != nil {
		p.errorf(p.curToken.Pos, "could not parse OFFSET value %s: %v", p.curToken.Literal, err)
		return nil
	}
	clause.Value = val
	return clause
}

func (p *Parser) expectPeek(t TokenType) bool {
	if p.peekTokenIs(t) {
		p.nextToken()
		return true
	}
	p.peekError(t)
	return false
}

func (p *Parser) expectPeekKeyword(kw string) bool {
	if p.peekTokenIsKeyword(kw) {
		p.nextToken()
		return true
	}
	p.errors = append(p.errors, fmt.Sprintf("expected next token to be keyword '%s', got %s (%s) at %s instead",
		kw, p.peekToken.Literal, p.peekToken.Type, p.peekToken.Pos.String()))
	return false
}

func (p *Parser) errorf(pos scanner.Position, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	p.errors = append(p.errors, fmt.Sprintf("Parser error at %s: %s", pos.String(), msg))
}

func (p *Parser) peekError(t TokenType) {
	p.errors = append(p.errors, fmt.Sprintf("expected next token to be %s, got %s (%s) at %s instead",
		t, p.peekToken.Type, p.peekToken.Literal, p.peekToken.Pos.String()))
}

func (p *Parser) curPrecedence() int {
	if p, ok := precedences[p.curToken.Type]; ok {
		return p
	}
	return LOWEST
}

func (p *Parser) peekPrecedence() int {
	if p, ok := precedences[p.peekToken.Type]; ok {
		return p
	}
	return LOWEST
}

type ParsedQuery struct {
	TableName        string
	Fields           []string          // Fields to select (e.g., ["id", "name"] or ["*"])
	KeywordQuery     string            // For full-text search part of WHERE
	VectorQueryText  string            // For vector similarity search part of WHERE
	Filters          []FilterCondition // For structured data filtering (e.g., age > 30 AND city = 'New York')
	OrderByField     string
	OrderByDirection string // "ASC" or "DESC"
	Limit            int64
	Offset           int64
	UseANN           bool // Whether to use Approximate Nearest Neighbor search
	K                int  // Number of nearest neighbors for vector search
}

// FilterCondition represents a single filter condition (e.g., field = value)
type FilterCondition struct {
	Field    string
	Operator string // e.g., "=", ">", "<", "LIKE"
	Value    interface{}
	// For more complex scenarios, we might need to support logical operators (AND, OR) here
	// or have a tree structure for filters.
}

// ConvertASTToParsedQuery converts a SelectStatement AST node into a ParsedQuery struct.
func ConvertASTToParsedQuery(stmt *SelectStatement) (*ParsedQuery, error) {
	if stmt == nil {
		return nil, fmt.Errorf("cannot convert nil statement to ParsedQuery")
	}
	pq := &ParsedQuery{
		Limit:  -1, // Default: no limit
		Offset: 0,  // Default: no offset
	}

	if stmt.From != nil {
		pq.TableName = stmt.From.Value
	} else {
		return nil, fmt.Errorf("FROM clause is missing")
	}

	for _, fieldExpr := range stmt.Fields {
		switch f := fieldExpr.(type) {
		case *StarExpression:
			pq.Fields = append(pq.Fields, "*")
		case *Identifier:
			pq.Fields = append(pq.Fields, f.Value)
		// TODO: Handle other expression types like function calls (COUNT(*)) if needed
		default:
			return nil, fmt.Errorf("unsupported field type in SELECT list: %T", f)
		}
	}

	if stmt.Where != nil {
		filters, keywordQuery, vectorQuery, err := extractConditions(stmt.Where)
		if err != nil {
			return nil, fmt.Errorf("error extracting WHERE conditions: %w", err)
		}
		pq.Filters = filters
		pq.KeywordQuery = keywordQuery
		pq.VectorQueryText = vectorQuery
		// TODO: Populate UseANN and K based on extracted conditions or specific syntax
	}

	if stmt.OrderBy != nil {
		if ident, ok := stmt.OrderBy.Field.(*Identifier); ok {
			pq.OrderByField = ident.Value
		} else {
			return nil, fmt.Errorf("ORDER BY field must be a simple identifier, got %T", stmt.OrderBy.Field)
		}
		pq.OrderByDirection = strings.ToUpper(stmt.OrderBy.Direction)
	}

	if stmt.Limit != nil {
		pq.Limit = stmt.Limit.Value
	}

	if stmt.Offset != nil {
		pq.Offset = stmt.Offset.Value
	}

	return pq, nil
}

// extractConditions recursively extracts filter conditions...
func extractConditions(expr Expression) (filters []FilterCondition, keywordQuery string, vectorQuery string, err error) {
	switch e := expr.(type) {
	case *BinaryExpression:
		// ... (BinaryExpression handling remains the same, but will benefit from GroupedExpression)
		leftIdent, leftIsIdent := e.Left.(*Identifier)

		if leftIsIdent && strings.ToLower(leftIdent.Value) == "_keyword_" && e.Operator == "=" {
			if lit, ok := e.Right.(*LiteralExpression); ok {
				if strVal, okStr := lit.Value.(string); okStr {
					keywordQuery = strVal
					return // Assume this is the only condition for now if it's a keyword query
				}
			}
		} else if leftIsIdent && strings.ToLower(leftIdent.Value) == "_vector_" && e.Operator == "=" {
			if lit, ok := e.Right.(*LiteralExpression); ok {
				if strVal, okStr := lit.Value.(string); okStr {
					vectorQuery = strVal
					return // Assume this is the only condition for now if it's a vector query
				}
			}
		} else if strings.ToUpper(e.Operator) == "AND" || strings.ToUpper(e.Operator) == "OR" {
			lFilters, lKeyword, lVector, lErr := extractConditions(e.Left)
			rFilters, rKeyword, rVector, rErr := extractConditions(e.Right)
			if lErr != nil {
				return nil, "", "", lErr
			}
			if rErr != nil {
				return nil, "", "", rErr
			}
			filters = append(lFilters, rFilters...)
			// Simplistic combination of keyword/vector queries (likely needs better logic)
			if lKeyword != "" {
				keywordQuery = lKeyword
			}
			if rKeyword != "" {
				keywordQuery = rKeyword
			} // This overwrites, needs AND/OR logic
			if lVector != "" {
				vectorQuery = lVector
			}
			if rVector != "" {
				vectorQuery = rVector
			} // This overwrites
			return
		} else if leftIsIdent {
			fc := FilterCondition{Field: leftIdent.Value, Operator: e.Operator}
			switch val := e.Right.(type) {
			case *LiteralExpression:
				fc.Value = val.Value
			case *Identifier:
				fc.Value = val.Value
			default:
				return nil, "", "", fmt.Errorf("unsupported right-hand side type in binary expression: %T", e.Right)
			}
			filters = append(filters, fc)
			return
		} else {
			return nil, "", "", fmt.Errorf("unsupported binary expression structure for WHERE: Left is %T", e.Left)
		}

	case *Identifier: // e.g., WHERE boolean_field (implicitly true)
		filters = append(filters, FilterCondition{Field: e.Value, Operator: "=", Value: true})
		return
	case *LiteralExpression: // e.g. WHERE true
		if boolVal, ok := e.Value.(bool); ok && boolVal {
			return
		} else if ok && !boolVal {
			return nil, "", "", fmt.Errorf("WHERE clause evaluates to constant false")
		}
		return nil, "", "", fmt.Errorf("unsupported literal expression in WHERE clause: %v", e.Value)

	case *GroupedExpression: // Handle parentheses
		// Recursively call extractConditions on the sub-expression
		// The results from the sub-expression are treated as a single unit
		// in the context of the outer expression. How these are combined
		// (e.g. if this grouped expression is part of an AND/OR) depends on the
		// calling context in the BinaryExpression case for AND/OR.
		// For now, we just pass through the extracted conditions from the sub-expression.
		// A more robust solution would involve building a condition tree.
		return extractConditions(e.SubExpression)

	default:
		return nil, "", "", fmt.Errorf("unsupported expression type in WHERE clause: %T", expr)
	}
	return
}

type QueryConditionNode interface {
	// Marker interface
}

type LogicalConditionNode struct {
	Operator string // "AND" or "OR"
	Left     QueryConditionNode
	Right    QueryConditionNode
}

type AtomicFilterCondition FilterCondition

func ParseQueryWithAST(query string) (*SelectStatement, *ParsedQuery, []string, error) {
	l := NewLexer(query)
	p := NewParser(l)

	stmtNode := p.ParseStatement()
	errs := p.Errors()
	if len(errs) > 0 {
		return nil, nil, errs, fmt.Errorf("parser has %d errors", len(errs))
	}

	if stmtNode == nil {
		return nil, nil, errs, fmt.Errorf("parsing returned nil statement without errors, this should not happen")
	}

	selectStmt, ok := stmtNode.(*SelectStatement)
	if !ok {
		return nil, nil, errs, fmt.Errorf("parsed statement is not a SelectStatement, got %T", stmtNode)
	}

	parsedParams, err := ConvertASTToParsedQuery(selectStmt)
	if err != nil {
		return selectStmt, nil, errs, fmt.Errorf("error converting AST to query parameters: %v", err)
	}

	return selectStmt, parsedParams, errs, nil // Return AST, parsed params, and any parser errors
}
