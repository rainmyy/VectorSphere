package common

/**
* this file const are golbal const for all data,import the package like "import . common"
 */
const (
	// DataType data file type,0,file default type
	DataType = iota
	// IniType ini file type, 1
	IniType
	// YamlType yaml file type, 2
	YamlType
	// JsonType json file type,3
	JsonType
)
const (
	// DataSuffix data file suffix
	DataSuffix = ".data"
	// IniSuffix ini file suffix
	IniSuffix = ".conf"
	// JsonSuffix json file suffix
	JsonSuffix = ".json"
	// YamlSuffix yaml file suffix
	YamlSuffix = ".yaml"
)
const (
	// LeftBracket left bracket, this data for turn tree data to string data
	LeftBracket rune = '['
	// RightBracket right bracket,this data for turn tree data to string data
	RightBracket rune = ']'
	// LeftRance right rance,this data for turn tree data to string data
	LeftRance  rune = '{'
	RightRance rune = '}'
	Colon      rune = ':'
	Comma      rune = ','
	None       rune = 'N'
	Slash      rune = '/'
	Hash       rune = '#'
	Asterisk   rune = '*'
	LineBreak  rune = '\n'
	Blank      rune = ' '
	Period     rune = '.'
)

const (
	BindTag = "bind"
	JsonTag = "json"
)
