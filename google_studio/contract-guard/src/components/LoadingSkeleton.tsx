'use client';

import { motion } from 'framer-motion';

interface LoadingSkeletonProps {
  className?: string;
  variant?: 'text' | 'card' | 'circle' | 'button';
  lines?: number;
}

export default function LoadingSkeleton({ 
  className = '', 
  variant = 'text',
  lines = 1 
}: LoadingSkeletonProps) {
  const baseClasses = 'loading-skeleton rounded';
  
  const variants = {
    text: 'h-4',
    card: 'h-32',
    circle: 'rounded-full aspect-square',
    button: 'h-10'
  };

  if (variant === 'text' && lines > 1) {
    return (
      <div className={`space-y-2 ${className}`}>
        {Array.from({ length: lines }).map((_, i) => (
          <motion.div
            key={i}
            className={`${baseClasses} ${variants.text} ${i === lines - 1 ? 'w-3/4' : 'w-full'}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.1 }}
          />
        ))}
      </div>
    );
  }

  return (
    <motion.div
      className={`${baseClasses} ${variants[variant]} ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    />
  );
}