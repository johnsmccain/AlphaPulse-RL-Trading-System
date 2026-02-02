'use client';

import { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Key, 
  X, 
  ArrowRight,
  FileSearch,
  Brain,
  AlertTriangle,
  CheckCircle,
  Sparkles,
  ExternalLink
} from 'lucide-react';

import ContractInput from '@/components/ContractInput';
import RiskSummary from '@/components/RiskSummary';
import FindingsList from '@/components/FindingsList';
import ExploitNarratives from '@/components/ExploitNarratives';
import AnalysisLoader from '@/components/AnalysisLoader';
import ErrorBoundary from '@/components/ErrorBoundary';
import MobileNav from '@/components/MobileNav';
import { useToast } from '@/components/Toast';

import { ContractAnalyzer } from '@/lib/gemini';
import { parseContract } from '@/lib/contract-parser';
import { AnalysisResult } from '@/lib/types';

export default function Home() {
  const [apiKey, setApiKey] = useState('');
  const [showApiKeyInput, setShowApiKeyInput] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<'findings' | 'narratives'>('findings');
  
  const { showToast } = useToast();

  const featurePills = useMemo(() => [
    { icon: FileSearch, text: 'Contract Parsing' },
    { icon: Brain, text: 'AI Risk Reasoning' },
    { icon: AlertTriangle, text: 'Exploit Narratives' },
    { icon: CheckCircle, text: 'Mitigation Suggestions' },
  ], []);

  const handleAnalyze = useCallback(async (sourceCode: string) => {
    // console.log(apiKey)
    if (!apiKey) {
      setError('Please enter your Gemini API key');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    try {
      // Step 1: Parse contract
      setCurrentStep('Parsing contract structure...');
      setAnalysisProgress(15);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const parsedContract = parseContract(sourceCode);
      
      // Step 2: Extract structure
      setCurrentStep('Extracting functions, variables, and access patterns...');
      setAnalysisProgress(35);
      await new Promise(resolve => setTimeout(resolve, 500));

      // Step 3: Analyze with Gemini
      setCurrentStep('Running Gemini AI analysis...');
      setAnalysisProgress(50);
      
      const analyzer = new ContractAnalyzer(apiKey);
      const analysisResult = await analyzer.analyzeContract(sourceCode, parsedContract);
      
      setAnalysisProgress(90);
      setCurrentStep('Generating report...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setAnalysisProgress(100);
      setCurrentStep('Analysis complete');
      setResult(analysisResult);
      
      showToast({
        type: 'success',
        title: 'Analysis Complete',
        message: `Found ${analysisResult.findings.length} security findings`,
      });
    } catch (err) {
      console.error('Analysis error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during analysis';
      setError(errorMessage);
      
      showToast({
        type: 'error',
        title: 'Analysis Failed',
        message: errorMessage,
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [apiKey]);

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    setAnalysisProgress(0);
    setCurrentStep('');
    
    showToast({
      type: 'info',
      title: 'Ready for New Analysis',
      message: 'You can now analyze another contract',
    });
  }, [showToast]);

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-linear-to-br from-primary/20 to-secondary/20 flex items-center justify-center border border-primary/30">
                <Shield className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white flex items-center gap-2">
                  ContractGuard
                  <span className="text-xs px-2 py-0.5 bg-primary/20 text-primary rounded-full border border-primary/30">
                    AI
                  </span>
                </h1>
                <p className="text-xs text-white/40">Smart Contract Risk Analysis</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Desktop API Key Input */}
              <div className="hidden md:flex items-center gap-4">
              {/* API Key Input */}
              {showApiKeyInput ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-2"
                >
                  <div className="relative">
                    <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                    <input
                      type="password"
                      placeholder="Enter Gemini API Key"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-black/50 border border-white/10 rounded-lg text-sm text-white placeholder-white/30 focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/20 w-64 transition-all"
                    />
                  </div>
                  {apiKey && (
                    <button
                      onClick={() => setShowApiKeyInput(false)}
                      className="px-4 py-2 bg-primary/20 text-primary text-sm font-medium rounded-lg hover:bg-primary/30 transition-colors border border-primary/30"
                    >
                      Save
                    </button>
                  )}
                </motion.div>
              ) : (
                <button
                  onClick={() => setShowApiKeyInput(true)}
                  className="text-sm text-white/40 hover:text-white transition-colors flex items-center gap-2"
                >
                  <Key className="w-4 h-4" />
                  Update API Key
                </button>
              )}

              <div className="h-6 w-px bg-white/10" />

              <a
                href="https://ai.google.dev/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-white/40 hover:text-white transition-colors flex items-center gap-1"
              >
                <Sparkles className="w-4 h-4" />
                Powered by Gemini
                <ExternalLink className="w-3 h-3" />
              </a>
              </div>

              {/* Mobile Navigation */}
              <MobileNav
                apiKey={apiKey}
                setApiKey={setApiKey}
                showApiKeyInput={showApiKeyInput}
                setShowApiKeyInput={setShowApiKeyInput}
              />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <ErrorBoundary>
          <AnimatePresence mode="wait">
          {!result && !isAnalyzing && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* Hero Section */}
              <div className="text-center mb-12">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 }}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full border border-white/10 mb-6"
                >
                  <Brain className="w-4 h-4 text-secondary" />
                  <span className="text-sm text-white/60">AI-Powered Security Analysis</span>
                </motion.div>

                <motion.h2
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="text-4xl md:text-5xl font-bold text-white mb-4"
                >
                  Understand{' '}
                  <span className="text-gradient">Smart Contract Risks</span>
                </motion.h2>

                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="text-lg text-white/50 max-w-2xl mx-auto mb-8"
                >
                  Paste your Solidity code and let Gemini AI analyze it for security vulnerabilities, 
                  generate exploit scenarios, and provide clear, actionable recommendations.
                </motion.p>

                {/* Feature Pills */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="flex flex-wrap justify-center gap-3 mb-12"
                >
                  {featurePills.map((feature) => (
                    <div
                      key={feature.text}
                      className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-lg border border-white/10"
                    >
                      <feature.icon className="w-4 h-4 text-primary" />
                      <span className="text-sm text-white/70">{feature.text}</span>
                    </div>
                  ))}
                </motion.div>
              </div>

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3"
                >
                  <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
                  <p className="text-red-400 text-sm">{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="ml-auto text-red-400 hover:text-red-300"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </motion.div>
              )}

              {/* Contract Input */}
              <ContractInput onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
            </motion.div>
          )}

          {isAnalyzing && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="py-20"
            >
              <AnalysisLoader currentStep={currentStep} progress={analysisProgress} />
            </motion.div>
          )}

          {result && !isAnalyzing && (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              {/* Results Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
                  <p className="text-sm text-white/40 mt-1">
                    Analyzed at {new Date(result.analyzedAt).toLocaleString()}
                  </p>
                </div>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-white/10 text-white text-sm font-medium rounded-lg hover:bg-white/20 transition-colors flex items-center gap-2 self-start sm:self-auto"
                >
                  <ArrowRight className="w-4 h-4 rotate-180" />
                  <span className="hidden sm:inline">Analyze Another Contract</span>
                  <span className="sm:hidden">New Analysis</span>
                </button>
              </div>

              {/* Risk Summary */}
              <RiskSummary result={result} />

              {/* Tab Navigation */}
              <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit overflow-x-auto">
                <button
                  onClick={() => setActiveResultTab('findings')}
                  className={`px-4 md:px-6 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                    activeResultTab === 'findings'
                      ? 'bg-primary/20 text-primary border border-primary/30'
                      : 'text-white/50 hover:text-white'
                  }`}
                >
                  Findings ({result.findings.length})
                </button>
                <button
                  onClick={() => setActiveResultTab('narratives')}
                  className={`px-4 md:px-6 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                    activeResultTab === 'narratives'
                      ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : 'text-white/50 hover:text-white'
                  }`}
                >
                  Attack Scenarios ({result.exploitNarratives.length})
                </button>
              </div>

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                {activeResultTab === 'findings' ? (
                  <motion.div
                    key="findings"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    <FindingsList findings={result.findings} />
                  </motion.div>
                ) : (
                  <motion.div
                    key="narratives"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                  >
                    <ExploitNarratives narratives={result.exploitNarratives} />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
          </AnimatePresence>
        </ErrorBoundary>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Shield className="w-5 h-5 text-primary/50" />
              <span className="text-sm text-white/30">
                ContractGuard © 2025. Built with Gemini AI.
              </span>
            </div>
            <div className="flex items-center gap-6">
              <span className="text-xs text-white/20">
                For educational purposes. Not a substitute for professional audits.
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
