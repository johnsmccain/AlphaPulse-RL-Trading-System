'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Shield, Key, Sparkles, ExternalLink } from 'lucide-react';

interface MobileNavProps {
  apiKey: string;
  setApiKey: (key: string) => void;
  showApiKeyInput: boolean;
  setShowApiKeyInput: (show: boolean) => void;
}

export default function MobileNav({ 
  apiKey, 
  setApiKey, 
  showApiKeyInput, 
  setShowApiKeyInput 
}: MobileNavProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden p-2 text-white/60 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50 rounded-lg"
        aria-label="Toggle mobile menu"
      >
        {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Mobile menu overlay */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 h-full w-80 bg-black/90 backdrop-blur-xl border-l border-white/10 z-50 md:hidden"
            >
              <div className="p-6 space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Shield className="w-6 h-6 text-primary" />
                    <span className="font-bold text-white">ContractGuard</span>
                  </div>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="p-2 text-white/60 hover:text-white transition-colors rounded-lg"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* API Key Section */}
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wide">
                    Configuration
                  </h3>
                  
                  {showApiKeyInput ? (
                    <div className="space-y-3">
                      <div className="relative">
                        <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                        <input
                          type="password"
                          placeholder="Enter Gemini API Key"
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          className="w-full pl-10 pr-4 py-3 bg-black/50 border border-white/10 rounded-lg text-sm text-white placeholder-white/30 focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/20"
                        />
                      </div>
                      {apiKey && (
                        <button
                          onClick={() => {
                            setShowApiKeyInput(false);
                            setIsOpen(false);
                          }}
                          className="w-full px-4 py-3 bg-primary/20 text-primary text-sm font-medium rounded-lg hover:bg-primary/30 transition-colors border border-primary/30"
                        >
                          Save API Key
                        </button>
                      )}
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowApiKeyInput(true)}
                      className="w-full text-left px-4 py-3 bg-white/5 text-white/60 hover:text-white hover:bg-white/10 transition-colors rounded-lg flex items-center gap-3"
                    >
                      <Key className="w-4 h-4" />
                      Update API Key
                    </button>
                  )}
                </div>

                {/* Links */}
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wide">
                    Resources
                  </h3>
                  
                  <a
                    href="https://ai.google.dev/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-full text-left px-4 py-3 bg-white/5 text-white/60 hover:text-white hover:bg-white/10 transition-colors rounded-lg flex items-center gap-3"
                    onClick={() => setIsOpen(false)}
                  >
                    <Sparkles className="w-4 h-4" />
                    Powered by Gemini
                    <ExternalLink className="w-3 h-3 ml-auto" />
                  </a>
                </div>

                {/* Footer */}
                <div className="pt-6 border-t border-white/10">
                  <p className="text-xs text-white/30 text-center">
                    ContractGuard © 2025
                    <br />
                    For educational purposes only
                  </p>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}