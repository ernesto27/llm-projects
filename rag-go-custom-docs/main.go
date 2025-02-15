package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type Document struct {
	Path     string
	Content  string
	Keywords map[string]int
}

type ChatRequest struct {
	Question string `json:"question"`
}

type ChatResponse struct {
	Answer string `json:"answer"`
	Error  string `json:"error,omitempty"`
}

func main() {
	searchParam := flag.String("search", "", "Search in documents and generate answer")
	port := flag.String("port", "8080", "Server port")
	flag.Parse()

	docs, err := loadDocuments("./docs")
	if err != nil {
		log.Fatal(err)
	}

	// If search parameter is provided, run in CLI mode
	if *searchParam != "" {
		runCLIMode(*searchParam, docs)
		return
	}

	// Otherwise, start HTTP server
	runServerMode(*port, docs)
}

func runCLIMode(searchParam string, docs []Document) {
	results := search(searchParam, docs)
	if len(results) == 0 {
		fmt.Println("No matching documents found")
		return
	}

	fmt.Printf("Most relevant document: %s\n", results[0].Path)
	fmt.Printf("Content preview:\n%s\n", truncateContent(results[0].Content, 300))

	fmt.Println("\nGenerating answer based on the document...")
	answer, err := queryOllama(results[0].Content, searchParam)
	if err != nil {
		log.Fatalf("Error querying Ollama: %v", err)
	}
	fmt.Printf("\nAnswer: %s\n", answer)
}

func runServerMode(port string, docs []Document) {
	http.HandleFunc("/", handleHome)
	http.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		handleChat(w, r, docs)
	})
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

	log.Printf("Server starting on port %s...", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}

func loadDocuments(root string) ([]Document, error) {
	var docs []Document
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && isTextFile(path) {
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			doc := Document{
				Path:     path,
				Content:  string(content),
				Keywords: extractKeywords(string(content)),
			}
			docs = append(docs, doc)
		}
		return nil
	})
	return docs, err
}

func isTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".txt" || ext == ".md" || ext == ".rst" || ext == ".json"
}

func extractKeywords(content string) map[string]int {
	words := strings.Fields(strings.ToLower(content))
	keywords := make(map[string]int)
	for _, word := range words {
		word = strings.Trim(word, ".,!?()[]{}\"'")
		if len(word) > 2 {
			keywords[word]++
		}
	}
	return keywords
}

func search(query string, docs []Document) []Document {
	queryWords := strings.Fields(strings.ToLower(query))
	scores := make([]float64, len(docs))

	for i, doc := range docs {
		var score float64
		for _, word := range queryWords {
			if count, exists := doc.Keywords[word]; exists {
				score += float64(count)
			}
		}
		scores[i] = score
	}

	// Sort documents by score
	var results []Document
	for i := range docs {
		if scores[i] > 0 {
			results = append(results, docs[i])
		}
	}

	// Simple bubble sort by score
	for i := 0; i < len(results)-1; i++ {
		for j := 0; j < len(results)-i-1; j++ {
			if scores[j] < scores[j+1] {
				results[j], results[j+1] = results[j+1], results[j]
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	return results
}

func truncateContent(content string, maxLen int) string {
	if len(content) <= maxLen {
		return content
	}
	return content[:maxLen] + "..."
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("templates/chat.html")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	tmpl.Execute(w, nil)
}

func handleChat(w http.ResponseWriter, r *http.Request, docs []Document) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var chatReq ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	results := search(chatReq.Question, docs)
	if len(results) == 0 {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Error: "No relevant information found to answer your question",
		})
		return
	}

	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Create a channel for streaming
	stream := make(chan string)
	errChan := make(chan error)

	// Start streaming in a goroutine
	go func() {
		err := streamOllama(results[0].Content, chatReq.Question, stream)
		if err != nil {
			errChan <- err
		}
		close(stream)
	}()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
		return
	}

	// Stream responses to client
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				return // Stream closed
			}
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		case err := <-errChan:
			fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
			flusher.Flush()
			return
		}
	}
}
