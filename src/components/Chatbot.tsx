import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Loader, Send } from 'lucide-react';
import styles from './Chatbot.module.css';

interface Message {
  text: string;
  isUser: boolean;
}

const Chatbot: React.FC = () => {
  const [input, setInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    const newMessage: Message = { text: input, isUser: true };
    setMessages(prev => [...prev, newMessage]);
    setInput('');

    try {
      const response = await axios.post<{ response: string }>('http://localhost:5000/chat', { message: input });

      console.log('Bot response:', response.data.response);

      const botMessage: Message = { text: response.data.response, isUser: false };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error communicating with chatbot:', error);
      const errorMessage: Message = { text: 'Sorry, I encountered an error. Please try again later.', isUser: false };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatbotContainer}>
      <header className={styles.header}>
        Celestial Chatbot
      </header>
      <div className={styles.messagesContainer}>
        {messages.map((message, index) => (
          <div key={index} className={styles.messageGroup}>
            <div className={`${styles.message} ${message.isUser ? styles.userMessage : styles.botMessage}`}>
              {message.isUser ? (
                message.text
              ) : (
                <div dangerouslySetInnerHTML={{ __html: message.text }} />
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about celestial bodies..."
          className={styles.input}
          disabled={isLoading}
          rows={3}
        />
        <button
          type="submit"
          disabled={isLoading}
          className={styles.button}
        >
          {isLoading ? (
            <Loader className={styles.loaderIcon} />
          ) : (
            <Send className={styles.sendIcon} />
          )}
        </button>
      </form>
    </div>
  );
}

export default Chatbot;