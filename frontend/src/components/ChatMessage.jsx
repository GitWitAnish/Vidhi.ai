import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ChevronDown,
  ChevronUp,
  FileText,
  Scale,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useTextToSpeech } from "../hooks/useTextToSpeech";
import "./ChatMessage.css";

function ChatMessage({ message }) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const { speak, stop, isSpeaking } = useTextToSpeech();
  const isUser = message.role === "user";

  const handleSpeak = () => {
    if (isSpeaking) {
      stop();
    } else {
      // Remove markdown formatting for cleaner speech
      const plainText = message.content
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .replace(/#{1,6}\s/g, "")
        .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
        .replace(/`/g, "");

      speak(plainText);
    }
  };

  // Function to convert plain text URLs to clickable links
  const renderTextWithLinks = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const parts = text.split(urlRegex);

    return parts.map((part, index) => {
      if (part.match(urlRegex)) {
        return (
          <a
            key={index}
            href={part}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-link"
          >
            {part}
          </a>
        );
      }
      return part;
    });
  };

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar">
        {isUser ? (
          "A"
        ) : (
          <img src="/logo.jpeg" alt="VIDHI.AI" className="avatar-logo" />
        )}
      </div>

      <div className="message-content">
        <ReactMarkdown
          components={{
            p: ({ children }) => <p>{children}</p>,
            text: ({ children }) => (
              <>{renderTextWithLinks(String(children))}</>
            ),
            code: ({ inline, children, ...props }) => {
              if (inline) {
                return <code {...props}>{children}</code>;
              }
              return (
                <pre>
                  <code {...props}>{children}</code>
                </pre>
              );
            },
            blockquote: ({ children }) => <blockquote>{children}</blockquote>,
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ),
          }}
        >
          {message.content || ""}
        </ReactMarkdown>

        {!isUser && (
          <div className="message-actions">
            <button
              className={`speak-button ${isSpeaking ? "speaking" : ""}`}
              onClick={handleSpeak}
              title={isSpeaking ? "Stop speaking" : "Read aloud"}
            >
              {isSpeaking ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>

            {message.sources && (
              <div className="sources-section">
                <button
                  className="sources-toggle"
                  onClick={() => setSourcesExpanded(!sourcesExpanded)}
                >
                  <FileText size={16} />
                  <span>Sources</span>
                  {sourcesExpanded ? (
                    <ChevronUp size={16} />
                  ) : (
                    <ChevronDown size={16} />
                  )}
                </button>

                {sourcesExpanded && (
                  <div className="sources-content">
                    {Array.isArray(message.sources) ? (
                      message.sources.map((source, idx) => (
                        <div key={idx} className="source-card">
                          <div className="source-header">
                            <FileText size={14} />
                            <span className="source-doc-name">
                              {source.document_name || "Unknown"}
                            </span>
                            {source.similarity_score != null && (
                              <span className="source-score">
                                {(source.similarity_score * 100).toFixed(1)}% match
                              </span>
                            )}
                          </div>
                          {source.text && (
                            <p className="source-text">{source.text}</p>
                          )}
                        </div>
                      ))
                    ) : (
                      <ReactMarkdown>{String(message.sources || "")}</ReactMarkdown>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatMessage;
