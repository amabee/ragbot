"use client";
import React, { useState } from "react";

const Header = ({ connectionStatus }) => {
  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected":
        return "bg-green-500";
      case "connecting":
        return "bg-yellow-500";
      case "disconnected":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };
  return (
    <div className="bg-gradient-to-r from-slate-800 to-slate-700 text-white p-6 shadow-lg">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-4xl animate-bounce">🤖</div>
            <div>
              <h1 className="text-3xl font-bold">StudyBot</h1>
              <p className="text-slate-200">
                Your friendly learning companion!
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}
            ></div>
            <span className="text-sm text-slate-200 capitalize">
              {connectionStatus}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Header;
