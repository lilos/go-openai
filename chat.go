package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf16"
)

// Chat message role defined by the OpenAI API.
const (
	ChatMessageRoleSystem    = "system"
	ChatMessageRoleUser      = "user"
	ChatMessageRoleAssistant = "assistant"
)

var (
	ErrChatCompletionInvalidModel       = errors.New("this model is not supported with this method, please use CreateCompletion client method instead") //nolint:lll
	ErrChatCompletionStreamNotSupported = errors.New("streaming is not supported with this method, please use CreateChatCompletionStream")              //nolint:lll
)

type Content string

func (c Content) MarshalJSON() ([]byte, error) {
	//return []byte(strconv.QuoteToASCII(string(c))), nil
	//https://github.com/golang/go/issues/39137
	var tmpStr strings.Builder
	tmpRune := []rune(c)
	iStart := 0
	for i, r := range tmpRune {
		if r >= 0x10000 {
			// 当前表情之前的
			if i > iStart {
				rt := strconv.QuoteToASCII(string(tmpRune[iStart:i]))
				tmpStr.WriteString(rt[1 : len(rt)-1])
			}
			// 当前表情
			a, b := utf16.EncodeRune(r)
			tmpStr.WriteString(fmt.Sprintf(`\u%x\u%x`, a, b))
			iStart = i + 1
		}
	}
	// 最后表情之后
	if iStart < len(tmpRune) {
		rt := strconv.QuoteToASCII(string(tmpRune[iStart:]))
		tmpStr.WriteString(rt[1 : len(rt)-1])
	}
	return []byte(tmpStr.String()), nil
}

type ChatCompletionMessage struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`

	// This property isn't in the official documentation, but it's in
	// the documentation for the official library for python:
	// - https://github.com/openai/openai-python/blob/main/chatml.md
	// - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
	Name string `json:"name,omitempty"`
}

// ChatCompletionRequest represents a request structure for chat completion API.
type ChatCompletionRequest struct {
	Model            string                  `json:"model"`
	Messages         []ChatCompletionMessage `json:"messages"`
	MaxTokens        int                     `json:"max_tokens,omitempty"`
	Temperature      float32                 `json:"temperature,omitempty"`
	TopP             float32                 `json:"top_p,omitempty"`
	N                int                     `json:"n,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Stop             []string                `json:"stop,omitempty"`
	PresencePenalty  float32                 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32                 `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int          `json:"logit_bias,omitempty"`
	User             string                  `json:"user,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

// ChatCompletionResponse represents a response structure for chat completion API.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

// CreateChatCompletion — API call to Create a completion for the chat message.
func (c *Client) CreateChatCompletion(
	ctx context.Context,
	request ChatCompletionRequest,
) (response ChatCompletionResponse, err error) {
	if request.Stream {
		err = ErrChatCompletionStreamNotSupported
		return
	}

	urlSuffix := "/chat/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrChatCompletionInvalidModel
		return
	}

	req, err := c.requestBuilder.build(ctx, http.MethodPost, c.fullURL(urlSuffix, request.Model), request)
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}
