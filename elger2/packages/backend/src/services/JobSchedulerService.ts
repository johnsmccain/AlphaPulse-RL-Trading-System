// Job Scheduler Service for the Hedera Execution & Payments Platform

import {
  Job,
  JobId,
  JobType,
  JobStatus,
  PayrollSchedule,
  ScheduleId,
  PayrollFrequency,
  ExecutionResult,
  JobRegistrationRequest,
  AuditEventType
} from '../types/index.js'
import { JobRepository, PayrollScheduleRepository } from '../database/repositories.js'
import { ExecutionPolicyEngineService } from './ExecutionPolicyEngineService.js'
import { IAuditLogService } from './AuditLogService.js'
import { v4 as uuidv4 } from 'uuid'

// ============================================================================
// Interfaces
// ============================================================================

export interface IJobSchedulerService {
  // Job management
  scheduleJob(job: JobRegistrationRequest, organizationId: string): Promise<JobId>
  cancelJob(jobId: JobId): Promise<boolean>
  getJobStatus(jobId: JobId): Promise<JobStatus | null>
  
  // Scheduling
  scheduleRecurring(schedule: PayrollSchedule): Promise<ScheduleId>
  updateSchedule(scheduleId: ScheduleId, schedule: Partial<PayrollSchedule>): Promise<boolean>
  
  // Execution
  executeJob(jobId: JobId): Promise<ExecutionResult>
  retryJob(jobId: JobId): Promise<ExecutionResult>
  
  // Queue management
  processScheduledJobs(): Promise<void>
  getQueueStatus(): Promise<QueueStatus>
}

export interface QueueStatus {
  pendingJobs: number
  scheduledJobs: number
  executingJobs: number
  totalJobs: number
}

export interface JobSchedulerServiceConfig {
  maxConcurrentJobs?: number
  jobRetryAttempts?: number
  jobTimeoutMs?: number
  scheduleCheckIntervalMs?: number
}

// ============================================================================
// Job Scheduler Service Implementation
// ============================================================================

export class JobSchedulerService implements IJobSchedulerService {
  private readonly maxConcurrentJobs: number
  private readonly jobRetryAttempts: number
  private readonly jobTimeoutMs: number
  private readonly scheduleCheckIntervalMs: number
  private currentExecutingJobs: Set<JobId> = new Set()
  private scheduleCheckTimer?: NodeJS.Timeout

  constructor(
    private readonly jobRepository: JobRepository,
    private readonly payrollScheduleRepository: PayrollScheduleRepository,
    private readonly executionPolicyEngine: ExecutionPolicyEngineService,
    private readonly auditLogService: IAuditLogService,
    config: JobSchedulerServiceConfig = {}
  ) {
    this.maxConcurrentJobs = config.maxConcurrentJobs ?? 10
    this.jobRetryAttempts = config.jobRetryAttempts ?? 3
    this.jobTimeoutMs = config.jobTimeoutMs ?? 30000 // 30 seconds
    this.scheduleCheckIntervalMs = config.scheduleCheckIntervalMs ?? 60000 // 1 minute

    // Start the schedule checker
    this.startScheduleChecker()
  }

  // ============================================================================
  // Job Management
  // ============================================================================

  async scheduleJob(jobRequest: JobRegistrationRequest, organizationId: string): Promise<JobId> {
    // First validate with the ExecutionPolicyEngine
    const isAuthorized = await this.executionPolicyEngine.isOrganizationAuthorized(organizationId)
    if (!isAuthorized) {
      throw new Error('Organization not authorized')
    }

    // Determine initial status and schedule time
    let status = JobStatus.PENDING
    let scheduledAt: Date | undefined

    if (jobRequest.jobData.cronExpression || jobRequest.jobData.nextExecution) {
      status = JobStatus.SCHEDULED
      scheduledAt = jobRequest.jobData.nextExecution || this.calculateNextExecution(jobRequest.jobData.cronExpression!)
    }

    // Create the job entity without id, createdAt, updatedAt (repository will add these)
    const jobData = {
      organizationId,
      type: jobRequest.jobType,
      status,
      policyId: jobRequest.policyId,
      executionData: jobRequest.jobData,
      payment: jobRequest.payment,
      scheduledAt
    }

    // Store the job in our repository
    const job = await this.jobRepository.create(jobData)

    // Log job creation
    await this.auditLogService.logEvent(AuditEventType.JOB_CREATED, organizationId, {
      jobId: job.id,
      jobType: jobRequest.jobType,
      status,
      scheduledAt: scheduledAt?.toISOString()
    })

    // If job is not scheduled, queue it for immediate execution
    if (status === JobStatus.PENDING) {
      await this.queueJobForExecution(job.id)
    }

    return job.id
  }

  async cancelJob(jobId: JobId): Promise<boolean> {
    const job = await this.jobRepository.findById(jobId)
    if (!job) return false

    // Can only cancel pending or scheduled jobs
    if (job.status !== JobStatus.PENDING && job.status !== JobStatus.SCHEDULED) {
      return false
    }

    await this.jobRepository.updateStatus(jobId, JobStatus.CANCELLED)

    // Log job cancellation
    await this.auditLogService.logEvent(AuditEventType.JOB_FAILED, job.organizationId, {
      jobId,
      reason: 'cancelled',
      previousStatus: job.status
    })

    return true
  }

  async getJobStatus(jobId: JobId): Promise<JobStatus | null> {
    const job = await this.jobRepository.findById(jobId)
    return job?.status ?? null
  }

  // ============================================================================
  // Recurring Schedule Management
  // ============================================================================

  async scheduleRecurring(schedule: PayrollSchedule): Promise<ScheduleId> {
    const scheduleId = uuidv4()
    const now = new Date()

    const newSchedule: PayrollSchedule = {
      ...schedule,
      id: scheduleId,
      createdAt: now,
      updatedAt: now
    }

    await this.payrollScheduleRepository.create(newSchedule)

    // Create initial payroll jobs for active employees
    if (schedule.isActive) {
      await this.createPayrollJobsForSchedule(newSchedule)
    }

    return scheduleId
  }

  async updateSchedule(scheduleId: ScheduleId, updates: Partial<PayrollSchedule>): Promise<boolean> {
    const schedule = await this.payrollScheduleRepository.findById(scheduleId)
    if (!schedule) return false

    const updatedSchedule = await this.payrollScheduleRepository.update(scheduleId, {
      ...updates,
      updatedAt: new Date()
    })

    return updatedSchedule !== null
  }

  // ============================================================================
  // Job Execution Coordination
  // ============================================================================

  async executeJob(jobId: JobId): Promise<ExecutionResult> {
    const job = await this.jobRepository.findById(jobId)
    if (!job) {
      return {
        success: false,
        jobId,
        error: 'Job not found'
      }
    }

    // Check if job can be executed
    if (job.status === JobStatus.EXECUTING) {
      return {
        success: false,
        jobId,
        error: 'Job is already executing'
      }
    }

    if (job.status === JobStatus.COMPLETED) {
      return {
        success: false,
        jobId,
        error: 'Job is already completed'
      }
    }

    if (job.status === JobStatus.CANCELLED) {
      return {
        success: false,
        jobId,
        error: 'Job has been cancelled'
      }
    }

    // Check concurrent job limit
    if (this.currentExecutingJobs.size >= this.maxConcurrentJobs) {
      return {
        success: false,
        jobId,
        error: 'Maximum concurrent jobs reached'
      }
    }

    try {
      // Execute the job workflow
      return await this.executeJobWorkflow(job)
    } catch (error) {
      await this.handleJobExecutionError(job, error)
      return {
        success: false,
        jobId,
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    }
  }

  private async executeJobWorkflow(job: Job): Promise<ExecutionResult> {
    // Phase 1: Pre-execution validation and setup
    await this.preExecutionPhase(job)

    // Phase 2: Policy validation
    const policyValidationResult = await this.policyValidationPhase(job)
    if (!policyValidationResult.success) {
      return policyValidationResult
    }

    // Phase 3: Job execution
    const executionResult = await this.jobExecutionPhase(job)

    // Phase 4: Post-execution handling
    await this.postExecutionPhase(job, executionResult)

    return executionResult
  }

  private async preExecutionPhase(job: Job): Promise<void> {
    // Mark job as executing
    await this.jobRepository.updateStatus(job.id, JobStatus.EXECUTING)
    this.currentExecutingJobs.add(job.id)

    // Log job execution start
    await this.auditLogService.logEvent(AuditEventType.JOB_EXECUTED, job.organizationId, {
      jobId: job.id,
      jobType: job.type,
      executionStarted: new Date().toISOString(),
      phase: 'pre-execution'
    })
  }

  private async policyValidationPhase(job: Job): Promise<ExecutionResult> {
    // Validate job execution with policy engine (Requirement 2.2)
    const isAuthorized = await this.executionPolicyEngine.validateExecution(
      job.id,
      'system', // executor
      job.payment.amount
    )

    if (!isAuthorized) {
      await this.jobRepository.updateStatus(job.id, JobStatus.FAILED)
      
      await this.auditLogService.logEvent(AuditEventType.JOB_FAILED, job.organizationId, {
        jobId: job.id,
        reason: 'policy_violation',
        error: 'Job execution not authorized by policy engine',
        phase: 'policy-validation'
      })

      return {
        success: false,
        jobId: job.id,
        error: 'Job execution not authorized by policy engine'
      }
    }

    return { success: true, jobId: job.id }
  }

  private async jobExecutionPhase(job: Job): Promise<ExecutionResult> {
    // Execute the job based on its type
    const executionResult = await this.executeJobByType(job)

    // Log execution phase completion
    await this.auditLogService.logEvent(AuditEventType.JOB_EXECUTED, job.organizationId, {
      jobId: job.id,
      phase: 'execution',
      success: executionResult.success,
      transactionId: executionResult.transactionId,
      error: executionResult.error
    })

    return executionResult
  }

  private async postExecutionPhase(job: Job, executionResult: ExecutionResult): Promise<void> {
    try {
      if (executionResult.success) {
        await this.handleJobSuccess(job, executionResult)
      } else {
        await this.handleJobFailure(job, executionResult)
      }
    } finally {
      // Always clean up
      this.currentExecutingJobs.delete(job.id)
    }
  }

  private async handleJobSuccess(job: Job, executionResult: ExecutionResult): Promise<void> {
    await this.jobRepository.updateStatus(job.id, JobStatus.COMPLETED)
    
    await this.auditLogService.logEvent(AuditEventType.JOB_COMPLETED, job.organizationId, {
      jobId: job.id,
      executionCompleted: new Date().toISOString(),
      transactionId: executionResult.transactionId,
      phase: 'post-execution'
    })

    // Handle recurring jobs
    if (job.type === JobType.PAYROLL && job.executionData.cronExpression) {
      await this.scheduleNextOccurrence(job)
    }
  }

  private async handleJobFailure(job: Job, executionResult: ExecutionResult): Promise<void> {
    await this.jobRepository.updateStatus(job.id, JobStatus.FAILED)
    
    await this.auditLogService.logEvent(AuditEventType.JOB_FAILED, job.organizationId, {
      jobId: job.id,
      reason: 'execution_failed',
      error: executionResult.error,
      phase: 'post-execution'
    })
  }

  private async handleJobExecutionError(job: Job, error: unknown): Promise<void> {
    await this.jobRepository.updateStatus(job.id, JobStatus.FAILED)
    
    await this.auditLogService.logEvent(AuditEventType.JOB_FAILED, job.organizationId, {
      jobId: job.id,
      reason: 'execution_error',
      error: error instanceof Error ? error.message : 'Unknown error',
      phase: 'error-handling'
    })

    // Clean up
    this.currentExecutingJobs.delete(job.id)
  }

  async retryJob(jobId: JobId): Promise<ExecutionResult> {
    const job = await this.jobRepository.findById(jobId)
    if (!job) {
      return {
        success: false,
        jobId,
        error: 'Job not found'
      }
    }

    // Only retry failed jobs
    if (job.status !== JobStatus.FAILED) {
      return {
        success: false,
        jobId,
        error: 'Only failed jobs can be retried'
      }
    }

    // Reset job status to pending and execute
    await this.jobRepository.updateStatus(jobId, JobStatus.PENDING)
    return this.executeJob(jobId)
  }

  // ============================================================================
  // Queue Management
  // ============================================================================

  async processScheduledJobs(): Promise<void> {
    const scheduledJobs = await this.jobRepository.findScheduledJobs()
    
    for (const job of scheduledJobs) {
      if (this.currentExecutingJobs.size >= this.maxConcurrentJobs) {
        break // Stop processing if we've reached the limit
      }

      // Execute the scheduled job directly
      await this.executeJob(job.id)
    }
  }

  private async queueJobForExecution(jobId: JobId): Promise<void> {
    // In a production system, this would add the job to a proper job queue
    // For testing, we'll just mark it as queued without executing immediately
    // The job will be executed when explicitly called via executeJob or processScheduledJobs
  }

  async getQueueStatus(): Promise<QueueStatus> {
    const [pendingJobs, scheduledJobs, executingJobs] = await Promise.all([
      this.jobRepository.findByStatus(JobStatus.PENDING),
      this.jobRepository.findByStatus(JobStatus.SCHEDULED),
      this.jobRepository.findByStatus(JobStatus.EXECUTING)
    ])

    return {
      pendingJobs: pendingJobs.length,
      scheduledJobs: scheduledJobs.length,
      executingJobs: executingJobs.length,
      totalJobs: pendingJobs.length + scheduledJobs.length + executingJobs.length
    }
  }

  // ============================================================================
  // Job Execution Coordination - Enhanced Workflow Management
  // ============================================================================

  /**
   * Enhanced job execution workflow with better error handling and coordination
   */
  async executeJobWithCoordination(jobId: JobId): Promise<ExecutionResult> {
    const job = await this.jobRepository.findById(jobId)
    if (!job) {
      return {
        success: false,
        jobId,
        error: 'Job not found'
      }
    }

    // Create execution context
    const executionContext = {
      jobId: job.id,
      organizationId: job.organizationId,
      startTime: new Date(),
      phase: 'initialization'
    }

    try {
      // Coordinate with other services based on job type
      const coordinationResult = await this.coordinateJobExecution(job, executionContext)
      return coordinationResult
    } catch (error) {
      await this.handleCoordinationError(job, executionContext, error)
      return {
        success: false,
        jobId,
        error: error instanceof Error ? error.message : 'Coordination error'
      }
    }
  }

  private async coordinateJobExecution(job: Job, context: any): Promise<ExecutionResult> {
    // Update context
    context.phase = 'coordination'

    // Log coordination start
    await this.auditLogService.logEvent(AuditEventType.JOB_EXECUTED, job.organizationId, {
      jobId: job.id,
      phase: 'coordination-start',
      jobType: job.type,
      timestamp: new Date().toISOString()
    })

    // Coordinate based on job type
    switch (job.type) {
      case JobType.AI_TASK:
        return await this.coordinateAITaskExecution(job, context)
      case JobType.PAYROLL:
        return await this.coordinatePayrollExecution(job, context)
      case JobType.VENDOR_PAYMENT:
        return await this.coordinateVendorPaymentExecution(job, context)
      case JobType.SCHEDULED:
        return await this.coordinateScheduledJobExecution(job, context)
      default:
        throw new Error(`Unsupported job type for coordination: ${job.type}`)
    }
  }

  private async coordinateAITaskExecution(job: Job, context: any): Promise<ExecutionResult> {
    // This would coordinate with AI Execution Service
    // For now, use the basic execution
    return await this.executeJob(job.id)
  }

  private async coordinatePayrollExecution(job: Job, context: any): Promise<ExecutionResult> {
    // This would coordinate with payroll processing systems
    // For now, use the basic execution
    return await this.executeJob(job.id)
  }

  private async coordinateVendorPaymentExecution(job: Job, context: any): Promise<ExecutionResult> {
    // This would coordinate with risk engine and payment systems
    // For now, use the basic execution
    return await this.executeJob(job.id)
  }

  private async coordinateScheduledJobExecution(job: Job, context: any): Promise<ExecutionResult> {
    // This would handle scheduled job coordination
    // For now, use the basic execution
    return await this.executeJob(job.id)
  }

  private async handleCoordinationError(job: Job, context: any, error: unknown): Promise<void> {
    await this.auditLogService.logEvent(AuditEventType.JOB_FAILED, job.organizationId, {
      jobId: job.id,
      phase: 'coordination-error',
      error: error instanceof Error ? error.message : 'Unknown coordination error',
      context: JSON.stringify(context)
    })
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private async executeJobByType(job: Job): Promise<ExecutionResult> {
    switch (job.type) {
      case JobType.AI_TASK:
        return this.executeAITask(job)
      case JobType.PAYROLL:
        return this.executePayrollJob(job)
      case JobType.VENDOR_PAYMENT:
        return this.executeVendorPayment(job)
      case JobType.SCHEDULED:
        return this.executeScheduledJob(job)
      default:
        return {
          success: false,
          jobId: job.id,
          error: `Unsupported job type: ${job.type}`
        }
    }
  }

  private async executeAITask(job: Job): Promise<ExecutionResult> {
    // This would integrate with the AI Execution Service
    // For now, return a mock success result
    return {
      success: true,
      jobId: job.id,
      transactionId: `tx_${uuidv4()}`
    }
  }

  private async executePayrollJob(job: Job): Promise<ExecutionResult> {
    // This would integrate with the payment processing system
    // For now, return a mock success result
    return {
      success: true,
      jobId: job.id,
      transactionId: `tx_${uuidv4()}`
    }
  }

  private async executeVendorPayment(job: Job): Promise<ExecutionResult> {
    // This would integrate with the payment processing system
    // For now, return a mock success result
    return {
      success: true,
      jobId: job.id,
      transactionId: `tx_${uuidv4()}`
    }
  }

  private async executeScheduledJob(job: Job): Promise<ExecutionResult> {
    // This would execute the scheduled job logic
    // For now, return a mock success result
    return {
      success: true,
      jobId: job.id,
      transactionId: `tx_${uuidv4()}`
    }
  }

  private async createPayrollJobsForSchedule(schedule: PayrollSchedule): Promise<void> {
    for (const employee of schedule.employees) {
      const jobId = uuidv4()
      const now = new Date()

      const job: Job = {
        id: jobId,
        organizationId: schedule.organizationId,
        type: JobType.PAYROLL,
        status: JobStatus.SCHEDULED,
        policyId: 'default-payroll-policy', // This should be configurable
        executionData: {
          employeeAccount: employee.hederaAccount,
          payrollPeriod: {
            startDate: now,
            endDate: schedule.nextPaymentDate,
            payPeriod: `${schedule.frequency}-${schedule.nextPaymentDate.toISOString().split('T')[0]}`
          }
        },
        payment: {
          amount: employee.salary,
          currency: employee.currency,
          recipient: employee.hederaAccount
        },
        createdAt: now,
        updatedAt: now,
        scheduledAt: schedule.nextPaymentDate
      }

      await this.jobRepository.create(job)
    }
  }

  private async scheduleNextOccurrence(job: Job): Promise<void> {
    if (!job.executionData.cronExpression) return

    const nextExecution = this.calculateNextExecution(job.executionData.cronExpression)
    
    // Create a new job for the next occurrence
    const nextJobId = uuidv4()
    const now = new Date()

    const nextJob: Job = {
      ...job,
      id: nextJobId,
      status: JobStatus.SCHEDULED,
      createdAt: now,
      updatedAt: now,
      scheduledAt: nextExecution,
      executedAt: undefined,
      completedAt: undefined
    }

    await this.jobRepository.create(nextJob)
  }

  private calculateNextExecution(cronExpression: string): Date {
    // This is a simplified implementation
    // In a real system, you'd use a proper cron parser like 'node-cron' or 'cron-parser'
    const now = new Date()
    
    // For demo purposes, just add 1 day
    return new Date(now.getTime() + 24 * 60 * 60 * 1000)
  }

  private startScheduleChecker(): void {
    this.scheduleCheckTimer = setInterval(async () => {
      try {
        await this.processScheduledJobs()
      } catch (error) {
        console.error('Error processing scheduled jobs:', error)
      }
    }, this.scheduleCheckIntervalMs)
  }

  public stopScheduleChecker(): void {
    if (this.scheduleCheckTimer) {
      clearInterval(this.scheduleCheckTimer)
      this.scheduleCheckTimer = undefined
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createJobSchedulerService(
  jobRepository: JobRepository,
  payrollScheduleRepository: PayrollScheduleRepository,
  executionPolicyEngine: ExecutionPolicyEngineService,
  auditLogService: IAuditLogService,
  config?: JobSchedulerServiceConfig
): IJobSchedulerService {
  return new JobSchedulerService(
    jobRepository,
    payrollScheduleRepository,
    executionPolicyEngine,
    auditLogService,
    config
  )
}