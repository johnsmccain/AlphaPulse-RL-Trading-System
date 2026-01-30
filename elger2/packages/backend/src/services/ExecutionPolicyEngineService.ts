import { Job, JobId, JobRegistrationRequest, HederaAccountId, PolicyId, JobStatus } from '../types/index.js'

/**
 * Service interface for interacting with the ExecutionPolicyEngine smart contract
 * This represents the contract interaction layer that would be implemented
 * to communicate with the deployed Hedera smart contract
 */
export interface ExecutionPolicyEngineService {
  /**
   * Register a new job with the ExecutionPolicyEngine contract
   * @param organizationAccount The organization's Hedera account ID
   * @param request The job registration request
   * @returns Promise resolving to the created job
   */
  registerJob(organizationAccount: HederaAccountId, request: JobRegistrationRequest): Promise<Job>

  /**
   * Get a job by its ID
   * @param jobId The job ID to retrieve
   * @returns Promise resolving to the job or null if not found
   */
  getJob(jobId: JobId): Promise<Job | null>

  /**
   * Check if an organization is authorized
   * @param organizationAccount The organization's account ID
   * @returns Promise resolving to authorization status
   */
  isOrganizationAuthorized(organizationAccount: HederaAccountId): Promise<boolean>

  /**
   * Get the next job ID that would be assigned
   * @returns Promise resolving to the next job ID
   */
  getNextJobId(): Promise<number>

  /**
   * Validate if a job execution is authorized by policies
   * @param jobId The job ID to validate
   * @param executor The executor account ID
   * @param amount The payment amount
   * @returns Promise resolving to authorization status
   */
  validateExecution(jobId: JobId, executor: string, amount: number): Promise<boolean>
}

/**
 * Mock implementation of ExecutionPolicyEngineService for testing
 * In production, this would be replaced with actual Hedera SDK integration
 */
export class MockExecutionPolicyEngineService implements ExecutionPolicyEngineService {
  private jobs: Map<JobId, Job> = new Map()
  private nextJobId = 1
  private authorizedOrganizations = new Set<HederaAccountId>()

  constructor() {
    // Add some default authorized organizations for testing
    this.authorizedOrganizations.add('0.0.1001')
    this.authorizedOrganizations.add('0.0.1002')
    this.authorizedOrganizations.add('0.0.1003')
  }

  async registerJob(organizationAccount: HederaAccountId, request: JobRegistrationRequest): Promise<Job> {
    // Validate organization is authorized
    if (!this.authorizedOrganizations.has(organizationAccount)) {
      throw new Error('Organization not authorized')
    }

    // Validate payment amount
    if (request.payment.amount <= 0) {
      throw new Error('Payment amount must be greater than 0')
    }

    // Validate payment recipient
    if (!request.payment.recipient || request.payment.recipient === '0.0.0') {
      throw new Error('Invalid payment recipient')
    }

    // Generate unique job ID
    const jobId = `job_${this.nextJobId++}`

    // Create job record with all required fields
    const job: Job = {
      id: jobId,
      organizationId: organizationAccount, // Associate with organization's treasury account
      type: request.jobType,
      status: JobStatus.PENDING,
      policyId: request.policyId,
      executionData: request.jobData,
      payment: request.payment,
      createdAt: new Date(),
      updatedAt: new Date()
    }

    // Store the job
    this.jobs.set(jobId, job)

    return job
  }

  async getJob(jobId: JobId): Promise<Job | null> {
    return this.jobs.get(jobId) || null
  }

  async isOrganizationAuthorized(organizationAccount: HederaAccountId): Promise<boolean> {
    return this.authorizedOrganizations.has(organizationAccount)
  }

  async getNextJobId(): Promise<number> {
    return this.nextJobId
  }

  async validateExecution(jobId: JobId, executor: string, amount: number): Promise<boolean> {
    const job = this.jobs.get(jobId)
    if (!job) {
      return false
    }

    // Basic validation - in a real implementation this would check policies
    if (amount <= 0) {
      return false
    }

    if (amount > 10000) { // Mock policy: max amount 10000
      return false
    }

    return true
  }

  // Test helper methods
  authorizeOrganization(organizationAccount: HederaAccountId): void {
    this.authorizedOrganizations.add(organizationAccount)
  }

  getAllJobs(): Job[] {
    return Array.from(this.jobs.values())
  }

  clear(): void {
    this.jobs.clear()
    this.nextJobId = 1
  }
}