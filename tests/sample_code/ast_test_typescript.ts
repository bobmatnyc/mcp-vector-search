#!/usr/bin/env ts-node
/**
 * Sample TypeScript file for testing AST enhancements.
 *
 * This file tests:
 * 1. Tree-sitter AST parsing for TypeScript
 * 2. Type annotation extraction
 * 3. Interface and type definition detection
 * 4. Generic type parameter extraction
 * 5. Decorator extraction (@deprecated, custom decorators)
 */

/**
 * User interface definition
 */
interface IUser {
    id: number;
    name: string;
    age: number;
    email?: string;
    roles: string[];
}

/**
 * Authentication result type
 */
type AuthResult = {
    success: boolean;
    token?: string;
    error?: string;
};

/**
 * Generic result wrapper
 */
type Result<T, E = Error> = {
    data?: T;
    error?: E;
};

/**
 * Simple function with type annotations (score = 1)
 */
function simpleGreeting(name: string): string {
    return `Hello, ${name}!`;
}

/**
 * Calculate letter grade with moderate complexity (score = 5)
 * @param score - Base score (0-100)
 * @param bonus - Bonus points to add
 * @returns Letter grade
 */
function calculateGrade(score: number, bonus: number = 0): string {
    const total: number = score + bonus;

    if (total >= 90) {  // +1
        return 'A';
    } else if (total >= 80) {  // +1
        return 'B';
    } else if (total >= 70) {  // +1
        return 'C';
    } else if (total >= 60) {  // +1
        return 'D';
    } else {
        return 'F';
    }
}

/**
 * Generic function with high complexity (score = 8)
 * @template T - Type of items to process
 * @param items - Array of items
 * @param predicate - Filter predicate
 * @param transform - Transform function
 * @returns Processed items
 */
function processItems<T, R>(
    items: T[],
    predicate: (item: T) => boolean,
    transform: (item: T) => R
): Result<R[], Error> {
    if (!items || !Array.isArray(items)) {  // +1
        return {
            error: new Error('Invalid items array')
        };
    }

    const results: R[] = [];

    try {  // +1
        for (const item of items) {  // +1
            if (!predicate(item)) {  // +1
                continue;
            }

            try {  // +1
                const transformed = transform(item);
                results.push(transformed);
            } catch (error) {
                console.error('Transform error:', error);
                continue;
            }
        }

        return { data: results };
    } catch (error) {
        return {
            error: error instanceof Error ? error : new Error(String(error))  // +1 (conditional)
        };
    }
}

// Arrow functions with type annotations

/**
 * Simple arrow function with types
 */
const doubleNumber = (x: number): number => x * 2;

/**
 * Arrow function returning typed promise (complexity = 3)
 */
const fetchUserData = async (userId: number): Promise<IUser | null> => {
    try {  // +1
        // Simulated async operation
        const response = await fetch(`/api/users/${userId}`);  // +1 (await)

        if (!response.ok) {  // +1
            return null;
        }

        return await response.json() as IUser;
    } catch (error) {
        console.error('Fetch error:', error);
        return null;
    }
};

/**
 * User class with TypeScript features (depth 1)
 *
 * Tests:
 * - Class with interface implementation
 * - Property type annotations
 * - Access modifiers (private, protected, public)
 * - Readonly properties
 * - Optional parameters
 */
class User implements IUser {
    readonly id: number;
    public name: string;
    public age: number;
    public email?: string;
    public roles: string[];
    private _password: string;
    protected _createdAt: Date;

    /**
     * Create a new user
     */
    constructor(
        id: number,
        name: string,
        age: number,
        password: string,
        email?: string,
        roles: string[] = ['user']
    ) {
        this.id = id;
        this.name = name;
        this.age = age;
        this._password = password;
        this.email = email;
        this.roles = roles;
        this._createdAt = new Date();
    }

    /**
     * Get formatted display name
     */
    get displayName(): string {
        return this.name.charAt(0).toUpperCase() + this.name.slice(1);
    }

    /**
     * Check if user is an adult (complexity = 2)
     */
    get isAdult(): boolean {
        if (this.age >= 18) {  // +1
            return true;
        }
        return false;
    }

    /**
     * Check if user has specific role (complexity = 3)
     */
    hasRole(role: string): boolean {
        if (!this.roles || this.roles.length === 0) {  // +1
            return false;
        }

        for (const userRole of this.roles) {  // +1
            if (userRole === role) {
                return true;
            }
        }

        return false;
    }

    /**
     * Validate email format (static method, complexity = 4)
     */
    static validateEmail(email: string): boolean {
        if (!email) {  // +1
            return false;
        }

        if (!email.includes('@')) {  // +1
            return false;
        }

        const parts = email.split('@');
        if (parts.length !== 2) {  // +1
            return false;
        }

        return true;
    }

    /**
     * Create user from object
     */
    static fromObject(data: Partial<IUser> & { password: string }): User {
        return new User(
            data.id || 0,
            data.name || '',
            data.age || 0,
            data.password,
            data.email,
            data.roles
        );
    }

    /**
     * Update user profile (complexity = 4)
     */
    updateProfile(updates: Partial<Omit<IUser, 'id'>>): void {
        if (updates.name !== undefined) {  // +1
            this.name = updates.name;
        }
        if (updates.age !== undefined) {  // +1
            this.age = updates.age;
        }
        if (updates.email !== undefined) {  // +1
            this.email = updates.email;
        }
    }

    /**
     * Verify password (private method, complexity = 2)
     */
    private verifyPassword(password: string): boolean {
        if (!password) {  // +1
            return false;
        }
        return this._password === password;
    }

    /**
     * Get user age in years (protected method)
     */
    protected getAgeInYears(): number {
        return this.age;
    }
}

/**
 * Abstract base class for testing inheritance
 */
abstract class BaseManager<T> {
    protected items: Map<string, T>;

    constructor() {
        this.items = new Map();
    }

    /**
     * Abstract method that must be implemented
     */
    abstract validate(item: T): boolean;

    /**
     * Add item with validation (complexity = 3)
     */
    add(key: string, item: T): Result<T, string> {
        if (!key) {  // +1
            return { error: 'Key is required' };
        }

        if (!this.validate(item)) {  // +1
            return { error: 'Validation failed' };
        }

        this.items.set(key, item);
        return { data: item };
    }

    /**
     * Get item by key (complexity = 2)
     */
    get(key: string): T | undefined {
        if (!this.items.has(key)) {  // +1
            return undefined;
        }
        return this.items.get(key);
    }
}

/**
 * Authentication manager with generics (depth 1)
 */
class AuthenticationManager extends BaseManager<IUser> {
    private secretKey: string;
    private timeout: number;

    constructor(secretKey: string, timeout: number = 3600) {
        super();
        this.secretKey = secretKey;
        this.timeout = timeout;
    }

    /**
     * Validate user implementation (complexity = 4)
     */
    validate(user: IUser): boolean {
        if (!user.name || user.name.length < 3) {  // +1
            return false;
        }

        if (user.age < 0 || user.age > 150) {  // +1
            return false;
        }

        if (user.email && !User.validateEmail(user.email)) {  // +1
            return false;
        }

        return true;
    }

    /**
     * Authenticate user and return token (high complexity = 8)
     */
    async authenticate(
        username: string,
        password: string,
        rememberMe: boolean = false
    ): Promise<AuthResult> {
        if (!username || !password) {  // +1
            return {
                success: false,
                error: 'Username and password required'
            };
        }

        // Validate password strength
        if (password.length < 8) {  // +1
            return {
                success: false,
                error: 'Password too short'
            };
        }

        try {  // +1
            // Simulate async credential validation
            const valid = await this.validateCredentials(username, password);

            if (!valid) {  // +1
                return {
                    success: false,
                    error: 'Invalid credentials'
                };
            }

            // Generate token
            const token = this.generateToken(username, rememberMe);

            return {
                success: true,
                token
            };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Authentication failed'  // +1 (conditional)
            };
        }
    }

    /**
     * Validate credentials (private async method, complexity = 4)
     */
    private async validateCredentials(username: string, password: string): Promise<boolean> {
        // Simulated async validation
        await new Promise(resolve => setTimeout(resolve, 100));

        if (!/[A-Z]/.test(password)) {  // +1
            return false;
        }

        if (!/[0-9]/.test(password)) {  // +1
            return false;
        }

        if (!/[!@#$%^&*]/.test(password)) {  // +1
            return false;
        }

        return true;
    }

    /**
     * Generate authentication token
     */
    private generateToken(username: string, rememberMe: boolean): string {
        const timeout = rememberMe ? this.timeout * 24 : this.timeout;  // +1
        const data = `${username}:${this.secretKey}:${Date.now()}:${timeout}`;

        // Simulated hash
        return Buffer.from(data).toString('base64');
    }
}

/**
 * Decorator example (if supported by TypeScript compiler)
 */
function deprecated(message: string) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        console.warn(`@deprecated: ${message}`);
        return descriptor;
    };
}

/**
 * Class with decorators
 */
class LegacyService {
    /**
     * Old method that should not be used
     * @deprecated Use newMethod instead
     */
    @deprecated('Use newMethod instead')
    oldMethod(): void {
        console.log('This is deprecated');
    }

    /**
     * New recommended method
     */
    newMethod(): void {
        console.log('This is the new way');
    }
}

// Export for testing
export {
    IUser,
    AuthResult,
    Result,
    simpleGreeting,
    calculateGrade,
    processItems,
    doubleNumber,
    fetchUserData,
    User,
    BaseManager,
    AuthenticationManager,
    LegacyService
};
