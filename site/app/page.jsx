"use client";
import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  User,
  BookOpen,
  Lightbulb,
  Calculator,
  Smile,
  Microscope,
  PenTool,
  Heart,
  ThumbsUp,
  Star,
  Rainbow,
  Sun,
  Moon,
} from "lucide-react";

// Typing animation component
const TypingText = ({ text, onComplete, speed = 30 }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    if (text.length === 0) {
      onComplete && onComplete();
      return;
    }

    let currentIndex = 0;
    setDisplayedText("");
    setIsTyping(true);

    const typingInterval = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayedText(text.slice(0, currentIndex + 1));
        currentIndex++;
      } else {
        setIsTyping(false);
        clearInterval(typingInterval);
        onComplete && onComplete();
      }
    }, speed);

    return () => clearInterval(typingInterval);
  }, [text, speed, onComplete]);

  return (
    <span>
      {displayedText}
      {isTyping && (
        <span className="inline-block w-2 h-4 bg-purple-400 ml-1 animate-pulse rounded-sm"></span>
      )}
    </span>
  );
};

export default function AliceChat() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi there! I'm Alice, your friendly AI assistant! 🌟 I love helping kids learn new things, solve problems, and have fun! What would you like to chat about today?",
      isUser: false,
      timestamp: new Date(),
      isTyping: false,
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const quickActions = [];

  const emojiReactions = ["😊", "🌟", "🎉", "❤️", "👍", "🤔", "😮", "🦄"];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getAliceResponse = async (userMessage) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BASE_URL}/v1/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: userMessage,
            book_threshold: process.env.NEXT_PUBLIC_BOOK_THRESHOLD,
            chunk_threshold: process.env.NEXT_PUBLIC_CHUNK_THRESHOLD,
            max_chunks: process.env.NEXT_PUBLIC_MAX_CHUNKS,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      return (
        data.response ||
        data.answer ||
        data.message ||
        "I'm having trouble understanding that right now, but I'm here to help! Can you try asking in a different way? 🌟"
      );
    } catch (error) {
      console.error("Backend connection error:", error);

      // Fallback responses when backend is unavailable
      const fallbackResponses = [
        "Oops! I'm having trouble connecting right now, but I'm still here to help! 😊 Can you try asking again?",
        "I'm having a little technical hiccup, but let's keep chatting! 🌟 What else would you like to know?",
        "My connection is a bit wobbly right now, but I'm still your friend Alice! 💫 Try asking me something else!",
        "Sorry, I'm having trouble hearing you clearly! 🦄 Can you ask me that again in a moment?",
      ];

      return fallbackResponses[
        Math.floor(Math.random() * fallbackResponses.length)
      ];
    }
  };

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: messageText,
      isUser: true,
      timestamp: new Date(),
      isTyping: false,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      // Get response from backend
      const botResponse = await getAliceResponse(messageText);

      const botMessageId = Date.now() + 1;
      const botMessage = {
        id: botMessageId,
        text: botResponse,
        isUser: false,
        timestamp: new Date(),
        isTyping: true, // Start with typing animation
      };

      setMessages((prev) => [...prev, botMessage]);
      setTypingMessageId(botMessageId);
      setIsLoading(false);
    } catch (error) {
      console.error("Error sending message:", error);

      // Show error message to user
      const errorMessageId = Date.now() + 1;
      const errorMessage = {
        id: errorMessageId,
        text: "Oops! Something went wrong. I'm still here to help though! 🌟 Try asking me again!",
        isUser: false,
        timestamp: new Date(),
        isTyping: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
      setTypingMessageId(errorMessageId);
      setIsLoading(false);
    }
  };

  const handleTypingComplete = (messageId) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, isTyping: false } : msg
      )
    );
    setTypingMessageId(null);
  };

  const handleQuickAction = (message) => {
    handleSendMessage(message);
  };

  const addEmoji = (emoji) => {
    setInputMessage((prev) => prev + emoji + " ");
    inputRef.current?.focus();
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputMessage);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 via-purple-50 to-blue-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 text-white shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-center space-x-3">
            <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
              <Sparkles className="w-7 h-7 text-purple-500" />
            </div>
            <div className="text-center">
              <h1 className="text-3xl font-bold">Alice</h1>
              <p className="text-purple-100">Your Friendly AI Helper! 🌟</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto p-4">
        <div className="bg-white rounded-3xl shadow-2xl border-4 border-purple-200 overflow-hidden">
          {/* Quick Action Buttons */}
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 border-b-4 border-purple-200">
            <h3 className="text-center text-lg font-bold text-purple-700 mb-4">
              ✨ What can Alice help you with today? ✨
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  className={`h-auto py-4 px-3 flex flex-col items-center space-y-2 rounded-2xl border-2 transition-all duration-200 transform hover:scale-105 font-medium ${action.color}`}
                  onClick={() => handleQuickAction(action.message)}
                >
                  <action.icon className="w-6 h-6" />
                  <span className="text-xs text-center">{action.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-6 bg-gradient-to-br from-blue-50/30 to-purple-50/30">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex items-start space-x-3 animate-in slide-in-from-bottom-2 duration-500 ${
                    message.isUser ? "flex-row-reverse space-x-reverse" : ""
                  }`}
                >
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
                      message.isUser
                        ? "bg-gradient-to-br from-blue-400 to-purple-400"
                        : "bg-gradient-to-br from-pink-400 to-purple-400"
                    }`}
                  >
                    {message.isUser ? (
                      <User className="w-5 h-5" />
                    ) : (
                      <Sparkles className="w-5 h-5" />
                    )}
                  </div>

                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl border-2 ${
                      message.isUser
                        ? "bg-gradient-to-br from-blue-400 to-purple-400 text-white border-blue-300 rounded-br-md"
                        : "bg-white text-gray-800 border-purple-200 rounded-bl-md shadow-md"
                    }`}
                  >
                    <div className="text-sm leading-relaxed font-medium">
                      {message.isTyping ? (
                        <TypingText
                          text={message.text}
                          onComplete={() => handleTypingComplete(message.id)}
                          speed={Math.random() * 20 + 20}
                        />
                      ) : (
                        <p>{message.text}</p>
                      )}
                    </div>
                    <p
                      className={`text-xs mt-2 ${
                        message.isUser ? "text-blue-100" : "text-gray-500"
                      }`}
                    >
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}

              {/* Alice Thinking Indicator */}
              {isLoading && (
                <div className="flex items-start space-x-3 animate-in slide-in-from-bottom-2 duration-300">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-400 to-purple-400 flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-white animate-spin" />
                  </div>
                  <div className="bg-white px-4 py-3 rounded-2xl rounded-bl-md border-2 border-purple-200 shadow-md">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-purple-700 font-medium">
                        Alice is thinking
                      </span>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-pink-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Section */}
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 border-t-4 border-purple-200">
            {/* Emoji Buttons */}
            {/* <div className="flex justify-center space-x-2 mb-4">
              {emojiReactions.map((emoji, index) => (
                <button
                  key={index}
                  className="w-10 h-10 rounded-full bg-white hover:bg-purple-100 border-2 border-purple-200 hover:border-purple-300 transition-all duration-200 hover:scale-110 shadow-sm"
                  onClick={() => addEmoji(emoji)}
                >
                  <span className="text-lg">{emoji}</span>
                </button>
              ))}
            </div> */}

            {/* Message Input */}
            <div className="flex space-x-3">
              <input
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message to Alice here!"
                maxLength={500}
                className="flex-1 px-4 py-3 rounded-2xl border-2 border-purple-300 focus:border-purple-500 focus:outline-none text-gray-700 placeholder-gray-400 font-medium"
                disabled={isLoading}
              />
              <button
                onClick={() => handleSendMessage(inputMessage)}
                disabled={isLoading || !inputMessage.trim()}
                className="bg-gradient-to-r from-purple-400 to-pink-400 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-300 disabled:to-gray-300 text-white px-6 py-3 rounded-2xl font-bold transition-all duration-200 hover:scale-105 disabled:hover:scale-100 shadow-lg"
              >
                <Send className="w-4 h-4 mr-2 inline" />
                Send
              </button>
            </div>

            {/* Character Counter and Instructions */}
            <div className="flex justify-between items-center mt-3 text-xs text-purple-600 font-medium">
              <span>✨ Press Enter to send your message! ✨</span>
              <span className="bg-purple-100 px-2 py-1 rounded-full">
                {inputMessage.length}/500
              </span>
            </div>
          </div>
        </div>

        {/* Fun Footer */}
        <div className="mt-6 text-center">
          <div className="inline-flex items-center space-x-2 bg-white px-4 py-2 rounded-full border-2 border-purple-200 shadow-lg">
            <Star className="w-4 h-4 text-yellow-400" />
            <span className="text-purple-700 font-medium text-sm">
              Alice loves helping kids learn and grow! 🌈
            </span>
            <Star className="w-4 h-4 text-yellow-400" />
          </div>
        </div>
      </div>
    </div>
  );
}
