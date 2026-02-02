'use client';

import { motion } from 'framer-motion';
import { Loader2, Shield, Code, Search, Sparkles, CheckCircle } from 'lucide-react';

interface AnalysisLoaderProps {
  currentStep: string;
  progress: number;
}

const steps = [
  { id: 'parsing', label: 'Parsing Contract', icon: Code },
  { id: 'extracting', label: 'Extracting Structure', icon: Search },
  { id: 'analyzing', label: 'AI Analysis', icon: Sparkles },
  { id: 'complete', label: 'Complete', icon: CheckCircle },
];

export default function AnalysisLoader({ currentStep, progress }: AnalysisLoaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-panel rounded-2xl p-8"
    >
      <div className="flex flex-col items-center">
        {/* Animated Shield */}
        <div className="relative mb-8">
          <motion.div
            className="w-24 h-24 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center"
            animate={{
              boxShadow: [
                '0 0 0 0 rgba(0, 255, 157, 0.4)',
                '0 0 0 20px rgba(0, 255, 157, 0)',
              ],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeOut',
            }}
            role="img"
            aria-label="Analysis in progress"
          >
            <Shield className="w-12 h-12 text-primary" />
          </motion.div>

          <motion.div
            className="absolute inset-0 rounded-full border-2 border-primary/30"
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
          />

          <motion.div
            className="absolute -inset-4 rounded-full border border-secondary/20"
            animate={{ rotate: -360 }}
            transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
          />
        </div>

        {/* Status Text */}
        <h3 className="text-xl font-bold text-white mb-2">Analyzing Contract</h3>
        <p className="text-white/50 mb-6">{currentStep || 'Initializing...'}</p>

        {/* Progress Bar */}
        <div className="w-full max-w-md mb-8">
          <div className="h-2 bg-white/10 rounded-full overflow-hidden" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
            <motion.div
              className="h-full bg-gradient-to-r from-primary to-secondary"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <div className="flex justify-between mt-2">
            <span className="text-xs text-white/40">Progress</span>
            <span className="text-xs text-primary font-mono">{progress}%</span>
          </div>
        </div>

        {/* Steps */}
        <div className="flex items-center gap-4">
          {steps.map((step, index) => {
            const StepIcon = step.icon;
            const isActive = currentStep.toLowerCase().includes(step.id) ||
              (step.id === 'analyzing' && currentStep.toLowerCase().includes('gemini'));
            const isPast = progress > (index + 1) * 25;

            return (
              <div key={step.id} className="flex items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isActive
                      ? 'bg-primary/20 border-2 border-primary text-primary'
                      : isPast
                      ? 'bg-primary/10 border border-primary/30 text-primary'
                      : 'bg-white/5 border border-white/10 text-white/30'
                  }`}
                >
                  {isActive ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <StepIcon className="w-4 h-4" />
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`w-8 h-0.5 mx-2 transition-colors duration-300 ${
                      isPast ? 'bg-primary/50' : 'bg-white/10'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Subtle Message */}
        <motion.p
          className="mt-8 text-sm text-white/30 text-center max-w-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          ContractGuard is using Gemini AI to analyze your smart contract for potential security risks...
        </motion.p>
      </div>
    </motion.div>
  );
}
