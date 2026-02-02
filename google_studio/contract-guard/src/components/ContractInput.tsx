'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Code2, FileText, Loader2, Sparkles } from 'lucide-react';
import { exampleContracts, ExampleContractKey } from '@/lib/example-contracts';

interface ContractInputProps {
  onAnalyze: (sourceCode: string) => void;
  isAnalyzing: boolean;
}

export default function ContractInput({ onAnalyze, isAnalyzing }: ContractInputProps) {
  const [sourceCode, setSourceCode] = useState('');
  const [activeTab, setActiveTab] = useState<'paste' | 'examples'>('paste');

  const handleAnalyze = () => {
    if (sourceCode.trim()) {
      onAnalyze(sourceCode);
    }
  };

  const loadExample = (key: ExampleContractKey) => {
    setSourceCode(exampleContracts[key].code);
    setActiveTab('paste');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-2xl overflow-hidden"
    >
      {/* Tabs */}
      <div className="flex border-b border-white/10">
        <button
          onClick={() => setActiveTab('paste')}
          className={`flex-1 px-6 py-4 flex items-center justify-center gap-2 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
            activeTab === 'paste'
              ? 'bg-gradient-to-r from-primary/20 to-transparent text-primary border-b-2 border-primary'
              : 'text-white/60 hover:text-white hover:bg-white/5'
          }`}
        >
          <Code2 className="w-4 h-4" />
          <span className="font-medium">Paste Code</span>
        </button>
        <button
          onClick={() => setActiveTab('examples')}
          className={`flex-1 px-6 py-4 flex items-center justify-center gap-2 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-secondary/50 ${
            activeTab === 'examples'
              ? 'bg-gradient-to-r from-secondary/20 to-transparent text-secondary border-b-2 border-secondary'
              : 'text-white/60 hover:text-white hover:bg-white/5'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span className="font-medium">Examples</span>
        </button>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'paste' ? (
          <div className="space-y-4">
            <textarea
              value={sourceCode}
              onChange={(e) => setSourceCode(e.target.value)}
              placeholder="// Paste your Solidity smart contract code here...
pragma solidity ^0.8.0;

contract YourContract {
    // ...
}"
              className="w-full h-80 bg-black/50 border border-white/10 rounded-xl p-4 text-sm font-mono text-white/90 placeholder-white/30 focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/20 resize-none transition-all"
              spellCheck={false}
              aria-label="Smart contract source code input"
            />

            <div className="flex items-center justify-between">
              <div className="text-sm text-white/40">
                {sourceCode.length > 0 && (
                  <span>{sourceCode.split('\n').length} lines</span>
                )}
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAnalyze}
                disabled={!sourceCode.trim() || isAnalyzing}
                className={`px-8 py-3 rounded-xl font-semibold flex items-center gap-3 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/50 ${
                  !sourceCode.trim() || isAnalyzing
                    ? 'bg-white/10 text-white/30 cursor-not-allowed'
                    : 'bg-gradient-to-r from-primary to-emerald-400 text-black hover:shadow-lg hover:shadow-primary/30'
                }`}
                aria-label={isAnalyzing ? 'Analysis in progress' : 'Start contract analysis'}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Analyze Contract
                  </>
                )}
              </motion.button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(Object.keys(exampleContracts) as ExampleContractKey[]).map((key) => (
              <motion.button
                key={key}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => loadExample(key)}
                className="p-5 bg-white/5 border border-white/10 rounded-xl text-left hover:border-primary/30 hover:bg-primary/5 transition-all duration-300 group"
              >
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:from-primary/30 group-hover:to-secondary/30 transition-all">
                    <FileText className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-white group-hover:text-primary transition-colors">
                      {exampleContracts[key].name}
                    </h3>
                    <p className="text-sm text-white/50 mt-1 line-clamp-2">
                      {exampleContracts[key].description}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
