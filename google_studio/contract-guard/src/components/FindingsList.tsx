'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertCircle, 
  AlertTriangle, 
  ChevronDown, 
  Code,
  Shield,
  ShieldAlert,
  Target,
  Wrench,
  Zap
} from 'lucide-react';
import { RiskFinding } from '@/lib/types';

interface FindingsListProps {
  findings: RiskFinding[];
}

const severityConfig = {
  LOW: {
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    icon: Shield,
  },
  MEDIUM: {
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    icon: AlertCircle,
  },
  HIGH: {
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon: AlertTriangle,
  },
  CRITICAL: {
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    icon: ShieldAlert,
  },
};

const categoryIcons = {
  'access-control': Shield,
  'fund-security': Zap,
  'logic': Code,
  'external-calls': Target,
  'upgradeability': Wrench,
};

export default function FindingsList({ findings }: FindingsListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Sort by severity - memoized for performance
  const sortedFindings = useMemo(() => {
    const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    return [...findings].sort((a, b) => order[a.severity] - order[b.severity]);
  }, [findings]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-2xl overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">Security Findings</h2>
        <span className="text-sm text-white/40">{findings.length} issues detected</span>
      </div>

      <div className="divide-y divide-white/5">
        {sortedFindings.map((finding, index) => {
          const config = severityConfig[finding.severity];
          const Icon = config.icon;
          const CategoryIcon = categoryIcons[finding.category] || Shield;
          const isExpanded = expandedId === finding.id;

          return (
            <motion.div
              key={finding.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`${config.bg} transition-colors`}
            >
              <button
                onClick={() => setExpandedId(isExpanded ? null : finding.id)}
                className="w-full px-6 py-4 flex items-start gap-4 text-left hover:bg-white/5 transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50"
                aria-expanded={isExpanded}
                aria-controls={`finding-${finding.id}`}
              >
                <div className={`mt-1 p-2 rounded-lg ${config.bg} ${config.border} border`}>
                  <Icon className={`w-4 h-4 ${config.color}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-1">
                    <span className={`text-xs font-medium px-2 py-0.5 rounded ${config.bg} ${config.color} ${config.border} border`}>
                      {finding.severity}
                    </span>
                    <span className="text-xs text-white/40 flex items-center gap-1">
                      <CategoryIcon className="w-3 h-3" />
                      {finding.category}
                    </span>
                  </div>
                  <h3 className="font-semibold text-white">{finding.title}</h3>
                  <p className="text-sm text-white/50 mt-1 line-clamp-2">{finding.description}</p>
                </div>

                <motion.div
                  animate={{ rotate: isExpanded ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                  className="mt-2"
                >
                  <ChevronDown className="w-5 h-5 text-white/30" />
                </motion.div>
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                    id={`finding-${finding.id}`}
                  >
                    <div className="px-6 pb-6 pt-2 space-y-4 ml-14">
                      {/* Exploit Scenario */}
                      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                        <h4 className="text-sm font-semibold text-orange-400 flex items-center gap-2 mb-2">
                          <Target className="w-4 h-4" />
                          Exploit Scenario
                        </h4>
                        <p className="text-sm text-white/70 leading-relaxed">
                          {finding.exploitScenario}
                        </p>
                      </div>

                      {/* Impact */}
                      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                        <h4 className="text-sm font-semibold text-red-400 flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4" />
                          Impact
                        </h4>
                        <p className="text-sm text-white/70 leading-relaxed">
                          {finding.impact}
                        </p>
                      </div>

                      {/* Affected Code */}
                      {finding.affectedCode && (
                        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                          <h4 className="text-sm font-semibold text-white/60 flex items-center gap-2 mb-2">
                            <Code className="w-4 h-4" />
                            Affected Code
                          </h4>
                          <pre className="text-xs text-white/60 font-mono overflow-x-auto">
                            {finding.affectedCode}
                          </pre>
                        </div>
                      )}

                      {/* Mitigation */}
                      <div className="bg-emerald-500/10 rounded-xl p-4 border border-emerald-500/30">
                        <h4 className="text-sm font-semibold text-emerald-400 flex items-center gap-2 mb-2">
                          <Wrench className="w-4 h-4" />
                          Mitigation
                        </h4>
                        <p className="text-sm text-white/70 leading-relaxed">
                          {finding.mitigation}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
