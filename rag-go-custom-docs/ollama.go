package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaResponse struct {
	Response string `json:"response"`
}

func savePromptToFile(prompt string) error {
	debugDir := "debug_prompts"
	if err := os.MkdirAll(debugDir, 0755); err != nil {
		return fmt.Errorf("error creating debug directory: %v", err)
	}

	filename := filepath.Join(debugDir, fmt.Sprintf("prompt_%s.txt", time.Now().Format("20060102_150405")))
	return os.WriteFile(filename, []byte(prompt), 0644)
}

func queryOllama(docContent, question string) (string, error) {
	promptTemplate := `Tu rol es ser un asistente tecnológico especializado de una empresa de desarrollo de software. Debes responder consultas de clientes potenciales utilizando exclusivamente la información proporcionada en el siguiente texto entre """. 

Tu objetivo es ayudar a entender nuestros servicios y capacidades técnicas de manera profesional y precisa. Si la información solicitada no está presente en la documentación, responde: "Lo siento, no tengo esa información en este momento".

"""
%s
"""

Consulta del cliente: %s`
	prompt := fmt.Sprintf(promptTemplate, docContent, question)

	// Save the generated prompt to a file
	if err := savePromptToFile(prompt); err != nil {
		return "", fmt.Errorf("error saving prompt: %v", err)
	}

	reqBody := struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
		Stream bool   `json:"stream"`
	}{
		Model:  "llama3.2:3b",
		Prompt: prompt,
		Stream: true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Initialize a final response string and stream decoder.
	var finalResponse string
	reader := bufio.NewReader(resp.Body)
	decoder := json.NewDecoder(reader)

	// Stream multiple JSON chunks.
	for {
		var chunk map[string]interface{}
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("error decoding response stream: %v", err)
		}
		if part, ok := chunk["response"].(string); ok {
			finalResponse += part
		}
	}

	return finalResponse, nil
}

func streamOllama(docContent, question string, stream chan<- string) error {
	promptTemplate := `Tu rol es ser un asistente tecnológico especializado de una empresa de desarrollo de software. Debes responder consultas de clientes potenciales utilizando exclusivamente la información proporcionada en el siguiente texto entre """. 

Tu objetivo es ayudar a entender nuestros servicios y capacidades técnicas de manera profesional y precisa. Si la información solicitada no está presente en la documentación, responde: "Lo siento, no tengo esa información en este momento".

"""
%s
"""

Consulta del cliente: %s`
	prompt := fmt.Sprintf(promptTemplate, docContent, question)

	reqBody := OllamaRequest{
		Model:  "llama3.2:3b",
		Prompt: prompt,
		Stream: true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	decoder := json.NewDecoder(reader)

	for {
		var chunk map[string]interface{}
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error decoding response stream: %v", err)
		}

		if response, ok := chunk["response"].(string); ok {
			stream <- response
		}

		// Check if done
		if done, ok := chunk["done"].(bool); ok && done {
			break
		}
	}

	return nil
}
