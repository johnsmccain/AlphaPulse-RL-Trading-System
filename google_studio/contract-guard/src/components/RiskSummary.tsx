'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Shield, ShieldAlert, ShieldCheck, ShieldX, TrendingUp } from 'lucide-react';
import { AnalysisResult } from '@/lib/types';

interface RiskSummaryProps {
  result: AnalysisResult;
}

const severityConfig = {
  LOW: {
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/20',
    border: 'border-emerald-500/30',
    icon: ShieldCheck,
    gradient: 'from-emerald-500 to-green-500',
  },
  MEDIUM: {
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/20',
    border: 'border-yellow-500/30',
    icon: Shield,
    gradient: 'from-yellow-500 to-orange-500',
  },
  HIGH: {
    color: 'text-orange-400',
    bg: 'bg-orange-500/20',
    border: 'border-orange-500/30',
    icon: ShieldAlert,
    gradient: 'from-orange-500 to-red-500',
  },
  CRITICAL: {
    color: 'text-red-400',
    bg: 'bg-red-500/20',
    border: 'border-red-500/30',
    icon: ShieldX,
    gradient: 'from-red-500 to-rose-600',
  },
};

export default function RiskSummary({ result }: RiskSummaryProps) {
  const config = severityConfig[result.riskLevel];
  // const Icon = config.icon;

  // Count findings by severity - memoized for performance
  const findingCounts = useMemo(() => ({
    CRITICAL: result.findings.filter((f) => f.severity === 'CRITICAL').length,
    HIGH: result.findings.filter((f) => f.severity === 'HIGH').length,
    MEDIUM: result.findings.filter((f) => f.severity === 'MEDIUM').length,
    LOW: result.findings.filter((f) => f.severity === 'LOW').length,
  }), [result.findings]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-2xl p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Risk Assessment</h2>
          <p className="text-sm text-white/50 mt-1">{result.contractName}</p>
        </div>
        <div className={`px-4 py-2 rounded-lg ${config.bg} ${config.border} border`}>
          <span className={`font-bold ${config.color}`}>{result.riskLevel}</span>
        </div>
      </div>

      {/* Score Gauge */}
      <div className="flex items-center gap-6">
        <div className="relative">
          <svg className="w-32 h-32 transform -rotate-90" role="img" aria-label={`Risk score: ${result.overallScore} out of 10`}>
            {/* Background circle */}
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-white/10"
            />
            {/* Progress circle */}
            <motion.circle
              cx="64"
              cy="64"
              r="56"
              stroke="url(#scoreGradient)"
              strokeWidth="8"
              fill="none"
              strokeLinecap="round"
              strokeDasharray={`${(result.overallScore / 10) * 351.86} 351.86`}
              initial={{ strokeDasharray: '0 351.86' }}
              animate={{ strokeDasharray: `${(result.overallScore / 10) * 351.86} 351.86` }}
              transition={{ duration: 1.5, ease: 'easeOut' }}
            />
            <defs>
              <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" className={`stop-${result.riskLevel === 'LOW' ? 'emerald' : result.riskLevel === 'MEDIUM' ? 'yellow' : result.riskLevel === 'HIGH' ? 'orange' : 'red'}-400`} stopColor={result.riskLevel === 'LOW' ? '#34d399' : result.riskLevel === 'MEDIUM' ? '#facc15' : result.riskLevel === 'HIGH' ? '#fb923c' : '#f87171'} />
                <stop offset="100%" className={`stop-${result.riskLevel === 'LOW' ? 'green' : result.riskLevel === 'MEDIUM' ? 'orange' : result.riskLevel === 'HIGH' ? 'red' : 'rose'}-500`} stopColor={result.riskLevel === 'LOW' ? '#22c55e' : result.riskLevel === 'MEDIUM' ? '#f97316' : result.riskLevel === 'HIGH' ? '#ef4444' : '#e11d48'} />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              className={`text-3xl font-bold ${config.color}`}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
            >
              {result.overallScore}
            </motion.span>
            <span className="text-xs text-white/40">/10</span>
          </div>
        </div>

        <div className="flex-1 space-y-3">
          <p className="text-white/70 text-sm leading-relaxed">{result.summary}</p>
          
          {/* Finding badges */}
          <div className="flex flex-wrap gap-2">
            {findingCounts.CRITICAL > 0 && (
              <span className="px-3 py-1 bg-red-500/20 text-red-400 text-xs font-medium rounded-full border border-red-500/30">
                {findingCounts.CRITICAL} Critical
              </span>
            )}
            {findingCounts.HIGH > 0 && (
              <span className="px-3 py-1 bg-orange-500/20 text-orange-400 text-xs font-medium rounded-full border border-orange-500/30">
                {findingCounts.HIGH} High
              </span>
            )}
            {findingCounts.MEDIUM > 0 && (
              <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 text-xs font-medium rounded-full border border-yellow-500/30">
                {findingCounts.MEDIUM} Medium
              </span>
            )}
            {findingCounts.LOW > 0 && (
              <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 text-xs font-medium rounded-full border border-emerald-500/30">
                {findingCounts.LOW} Low
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Positive Aspects */}
      {result.positiveAspects.length > 0 && (
        <div className="pt-4 border-t border-white/10">
          <h3 className="text-sm font-semibold text-emerald-400 flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4" />
            Positive Aspects
          </h3>
          <ul className="space-y-2">
            {result.positiveAspects.map((aspect, index) => (
              <li key={index} className="text-sm text-white/60 flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✓</span>
                {aspect}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Quick Recommendations */}
      {result.recommendations.length > 0 && (
        <div className="pt-4 border-t border-white/10">
          <h3 className="text-sm font-semibold text-secondary flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4" />
            Key Recommendations
          </h3>
          <ul className="space-y-2">
            {result.recommendations.slice(0, 3).map((rec, index) => (
              <li key={index} className="text-sm text-white/60 flex items-start gap-2">
                <span className="text-secondary mt-0.5">→</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}
