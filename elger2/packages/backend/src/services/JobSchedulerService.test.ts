// Tests for JobSchedulerService

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { JobSchedulerService, createJobSchedulerService } from './JobSchedulerService.js'
import { MockExecutionPolicyEngineService } from './ExecutionPolicyEngineService.js'
import { createAuditLogService } from './AuditLogService.js'
import { MemoryDatabase } from '../database/memory.js'
import { RepositoryFactory } from '../database/repositories.js'
import {
  JobType,
  JobStatus,
  PayrollFrequency,
  JobRegistrationRequest,
  PayrollSchedule
} from '../types/index.js'

// Mock HederaTransactionService for testing
const mockHederaService = {
  submitMessage: vi.fn().mockResolvedValue('mock-message-id'),
  createTopic: vi.fn().mockResolvedValue('mock-topic-id'),
  callContract: vi.fn(),
  deployContract: vi.fn(),
  transferTokens: vi.fn(),
  createToken: vi.fn(),
  createAccount: vi.fn(),
  getAccountBalance: vi.fn()
}

describe('JobSchedulerService', () => {
  let jobScheduler: JobSchedulerService
  let db: MemoryDatabase
  let repositoryFactory: RepositoryFactory
  let mockPolicyEngine: MockExecutionPolicyEngineService

  beforeEach(async () => {
    db = new MemoryDatabase()
    repositoryFactory = new RepositoryFactory(db)
    mockPolicyEngine = new MockExecutionPolicyEngineService()
    
    // Authorize the test organization
    mockPolicyEngine.authorizeOrganization('org-1')
    
    const auditLogService = createAuditLogService(
      mockHederaService as any,
      { auditTopicId: 'test-topic' }
    )

    jobScheduler = new JobSchedulerService(
      repositoryFactory.createJobRepository(),
      repositoryFactory.createPayrollScheduleRepository(),
      mockPolicyEngine,
      auditLogService,
      {
        maxConcurrentJobs: 5,
        jobRetryAttempts: 2,
        jobTimeoutMs: 5000,
        scheduleCheckIntervalMs: 1000
      }
    )
  })

  afterEach(() => {
    jobScheduler.stopScheduleChecker()
  })

  describe('scheduleJob', () => {
    it('should schedule an AI task job', async () => {
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: {
          taskDescription: 'Test AI task',
          requiredCapabilities: ['nlp', 'analysis'],
          expectedOutput: 'Analysis report'
        },
        policyId: 'policy-1',
        payment: {
          amount: 100,
          currency: 'HBAR',
          recipient: '0.0.12345'
        }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      
      expect(jobId).toBeDefined()
      expect(typeof jobId).toBe('string')

      // Add a small delay to ensure async operations complete
      await new Promise(resolve => setTimeout(resolve, 10))

      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.PENDING)
    })

    it('should schedule a job with future execution time', async () => {
      const futureDate = new Date(Date.now() + 60000) // 1 minute from now
      
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.SCHEDULED,
        jobData: {
          nextExecution: futureDate,
          taskDescription: 'Scheduled task'
        },
        policyId: 'policy-1',
        payment: {
          amount: 50,
          currency: 'HBAR',
          recipient: '0.0.12345'
        }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      const status = await jobScheduler.getJobStatus(jobId)
      
      expect(status).toBe(JobStatus.SCHEDULED)
    })

    it('should schedule a payroll job', async () => {
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.PAYROLL,
        jobData: {
          employeeAccount: '0.0.54321',
          payrollPeriod: {
            startDate: new Date('2024-01-01'),
            endDate: new Date('2024-01-31'),
            payPeriod: 'January 2024'
          }
        },
        policyId: 'payroll-policy-1',
        payment: {
          amount: 5000,
          currency: 'HBAR',
          recipient: '0.0.54321'
        }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      
      expect(jobId).toBeDefined()
      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.PENDING)
    })
  })

  describe('cancelJob', () => {
    it('should cancel a pending job', async () => {
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      const cancelled = await jobScheduler.cancelJob(jobId)
      
      expect(cancelled).toBe(true)
      
      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.CANCELLED)
    })

    it('should not cancel a non-existent job', async () => {
      const cancelled = await jobScheduler.cancelJob('non-existent-job')
      expect(cancelled).toBe(false)
    })
  })

  describe('executeJob', () => {
    it('should execute a valid job successfully', async () => {
      // Mock policy engine to authorize execution
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(true)

      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      const result = await jobScheduler.executeJob(jobId)
      
      expect(result.success).toBe(true)
      expect(result.jobId).toBe(jobId)
      expect(result.transactionId).toBeDefined()

      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.COMPLETED)
    })

    it('should fail job execution when policy validation fails', async () => {
      // Mock policy engine to reject execution
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(false)

      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      const result = await jobScheduler.executeJob(jobId)
      
      expect(result.success).toBe(false)
      expect(result.error).toContain('not authorized')

      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.FAILED)
    })

    it('should not execute a non-existent job', async () => {
      const result = await jobScheduler.executeJob('non-existent-job')
      
      expect(result.success).toBe(false)
      expect(result.error).toBe('Job not found')
    })
  })

  describe('retryJob', () => {
    it('should retry a failed job', async () => {
      // First, create and fail a job
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(false)

      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      await jobScheduler.executeJob(jobId) // This will fail

      // Now allow the retry to succeed
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(true)

      const retryResult = await jobScheduler.retryJob(jobId)
      
      expect(retryResult.success).toBe(true)
      
      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.COMPLETED)
    })

    it('should not retry a completed job', async () => {
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(true)

      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      await jobScheduler.executeJob(jobId) // This will succeed

      const retryResult = await jobScheduler.retryJob(jobId)
      
      expect(retryResult.success).toBe(false)
      expect(retryResult.error).toContain('Only failed jobs can be retried')
    })
  })

  describe('scheduleRecurring', () => {
    it('should create a recurring payroll schedule', async () => {
      const schedule: PayrollSchedule = {
        id: '', // Will be generated
        organizationId: 'org-1',
        name: 'Monthly Payroll',
        employees: [
          {
            employeeId: 'emp-1',
            hederaAccount: '0.0.11111',
            salary: 5000,
            currency: 'HBAR',
            startDate: new Date('2024-01-01')
          },
          {
            employeeId: 'emp-2',
            hederaAccount: '0.0.22222',
            salary: 6000,
            currency: 'HBAR',
            startDate: new Date('2024-01-01')
          }
        ],
        frequency: PayrollFrequency.MONTHLY,
        nextPaymentDate: new Date('2024-02-01'),
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date()
      }

      const scheduleId = await jobScheduler.scheduleRecurring(schedule)
      
      expect(scheduleId).toBeDefined()
      expect(typeof scheduleId).toBe('string')
    })
  })

  describe('getQueueStatus', () => {
    it('should return current queue status', async () => {
      // Create some jobs in different states
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.AI_TASK,
        jobData: { taskDescription: 'Test task' },
        policyId: 'policy-1',
        payment: { amount: 100, currency: 'HBAR', recipient: '0.0.12345' }
      }

      await jobScheduler.scheduleJob(jobRequest, 'org-1')
      await jobScheduler.scheduleJob(jobRequest, 'org-1')

      const status = await jobScheduler.getQueueStatus()
      
      expect(status.pendingJobs).toBeGreaterThanOrEqual(0)
      expect(status.scheduledJobs).toBeGreaterThanOrEqual(0)
      expect(status.executingJobs).toBeGreaterThanOrEqual(0)
      expect(status.totalJobs).toBeGreaterThanOrEqual(2)
    })
  })

  describe('processScheduledJobs', () => {
    it('should process scheduled jobs that are due', async () => {
      vi.spyOn(mockPolicyEngine, 'validateExecution').mockResolvedValue(true)

      // Create a scheduled job that's due now
      const pastDate = new Date(Date.now() - 1000) // 1 second ago
      
      const jobRequest: JobRegistrationRequest = {
        jobType: JobType.SCHEDULED,
        jobData: {
          nextExecution: pastDate,
          taskDescription: 'Scheduled task'
        },
        policyId: 'policy-1',
        payment: { amount: 50, currency: 'HBAR', recipient: '0.0.12345' }
      }

      const jobId = await jobScheduler.scheduleJob(jobRequest, 'org-1')
      
      // Process scheduled jobs
      await jobScheduler.processScheduledJobs()
      
      // Job should now be completed
      const status = await jobScheduler.getJobStatus(jobId)
      expect(status).toBe(JobStatus.COMPLETED)
    })
  })
})