#!/usr/bin/env node
/**
 * Sample JavaScript file for testing AST enhancements.
 *
 * This file tests:
 * 1. Tree-sitter AST parsing (not regex fallback)
 * 2. Complexity score calculation for JavaScript
 * 3. Function, arrow function, and class detection
 * 4. JSDoc comment extraction
 * 5. Hierarchical chunk relationships (module → class → method)
 */

/**
 * Simple function with low complexity (score = 1)
 * @param {string} name - The name to greet
 * @returns {string} Greeting message
 */
function simpleGreeting(name) {
    return `Hello, ${name}!`;
}

/**
 * Calculate letter grade with moderate complexity (score = 5)
 * @param {number} score - Base score (0-100)
 * @param {number} bonus - Bonus points to add (default: 0)
 * @returns {string} Letter grade (A, B, C, D, F)
 */
function calculateGrade(score, bonus = 0) {
    const total = score + bonus;

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
 * Validate data with high complexity (score = 10+)
 * @param {Object} data - Data to validate
 * @returns {Array<string>|null} Validation errors or null
 */
function complexValidator(data) {
    const errors = [];

    if (!data) {  // +1
        return ['Data is empty'];
    }

    if (!data.name) {  // +1
        errors.push('Missing name field');
    } else if (data.name.length < 3) {  // +1
        errors.push('Name too short');
    }

    if (data.age !== undefined) {  // +1
        try {  // +1
            const age = parseInt(data.age);
            if (age < 0 || age > 150) {  // +1
                errors.push('Invalid age range');
            }
        } catch (e) {
            errors.push('Age must be a number');
        }
    }

    if (data.tags) {  // +1
        for (const tag of data.tags) {  // +1
            if (typeof tag !== 'string') {  // +1
                errors.push(`Invalid tag type: ${tag}`);
            }
        }
    }

    return errors.length > 0 ? errors : null;  // +1 (conditional)
}

// Arrow functions with varying complexity

/**
 * Simple arrow function (score = 1)
 */
const doubleNumber = (x) => x * 2;

/**
 * Arrow function with moderate complexity (score = 3)
 * @param {Array<number>} numbers - Array of numbers
 * @returns {Array<number>} Filtered and doubled numbers
 */
const processNumbers = (numbers) => {
    return numbers
        .filter(n => n > 0)  // +1 (arrow function)
        .map(n => n * 2)     // +1 (arrow function)
        .sort((a, b) => a - b);  // +1 (arrow function)
};

/**
 * Arrow function with high complexity (score = 7)
 * @param {Array<Object>} users - Array of user objects
 * @param {number} minAge - Minimum age filter
 * @returns {Array<Object>} Filtered and processed users
 */
const filterAndProcessUsers = (users, minAge = 18) => {
    const results = [];

    if (!users || !Array.isArray(users)) {  // +1
        return results;
    }

    for (const user of users) {  // +1
        if (!user.age) {  // +1
            continue;
        }

        if (user.age < minAge) {  // +1
            continue;
        }

        if (user.active === undefined) {  // +1
            user.active = true;
        }

        results.push({
            ...user,
            displayName: user.name ? user.name.toUpperCase() : 'UNKNOWN'  // +1 (conditional)
        });
    }

    return results;
};

/**
 * User class with various method types (depth 1, parent = module)
 *
 * Tests:
 * - Class chunk extraction
 * - Method chunks (depth 2, parent = User class)
 * - Constructor detection
 * - Static method detection
 */
class User {
    /**
     * Create a new user
     * @param {string} name - User's name
     * @param {number} age - User's age
     * @param {string} email - User's email (optional)
     */
    constructor(name, age, email = null) {
        this.name = name;
        this.age = age;
        this.email = email;
        this._cache = new Map();
    }

    /**
     * Get formatted display name
     * @returns {string} Formatted name
     */
    get displayName() {
        return this.name.charAt(0).toUpperCase() + this.name.slice(1);
    }

    /**
     * Check if user is an adult (complexity = 2)
     * @returns {boolean} True if adult
     */
    get isAdult() {
        if (this.age >= 18) {  // +1
            return true;
        }
        return false;
    }

    /**
     * Validate email format (moderate complexity)
     * @param {string} email - Email to validate
     * @returns {boolean} True if valid
     */
    static validateEmail(email) {
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
     * @param {Object} data - User data
     * @returns {User} New user instance
     */
    static fromObject(data) {
        return new User(
            data.name || '',
            data.age || 0,
            data.email
        );
    }

    /**
     * Update user profile with optional fields (complexity = 4)
     * @param {Object} updates - Fields to update
     */
    updateProfile(updates) {
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
     * Get cached value or compute (complexity = 3)
     * @param {string} key - Cache key
     * @param {Function} computeFn - Function to compute value
     * @returns {*} Cached or computed value
     */
    getCached(key, computeFn) {
        if (this._cache.has(key)) {  // +1
            return this._cache.get(key);
        }

        try {  // +1
            const value = computeFn();
            this._cache.set(key, value);
            return value;
        } catch (error) {
            console.error('Compute error:', error);
            return null;
        }
    }
}

/**
 * Authentication manager class with complex methods (depth 1)
 *
 * Tests hierarchical relationships with nested complexity.
 */
class AuthenticationManager {
    /**
     * Initialize authentication manager
     * @param {string} secretKey - Secret key for tokens
     * @param {number} timeout - Token timeout in seconds
     */
    constructor(secretKey, timeout = 3600) {
        this.secretKey = secretKey;
        this.timeout = timeout;
        this._cache = new Map();
    }

    /**
     * Authenticate user and return token (high complexity = 8)
     * @param {string} username - Username
     * @param {string} password - Password
     * @param {boolean} rememberMe - Extend token lifetime
     * @returns {string|null} Token or null if failed
     */
    authenticate(username, password, rememberMe = false) {
        if (!username || !password) {  // +1
            return null;
        }

        // Check cache first
        if (this._cache.has(username)) {  // +1
            const cached = this._cache.get(username);
            if (cached.password === password) {  // +1
                if (rememberMe) {  // +1
                    cached.timeout = this.timeout * 24;  // 24 hours
                }
                return cached.token;
            }
        }

        // Validate credentials
        let valid = false;
        try {  // +1
            valid = this._validateCredentials(username, password);
        } catch (error) {
            console.error('Validation error:', error);
            return null;
        }

        if (!valid) {  // +1
            return null;
        }

        // Generate token
        const token = this._generateToken(username);

        // Cache result
        this._cache.set(username, {
            password,
            token,
            timeout: rememberMe ? this.timeout * 24 : this.timeout  // +1 (conditional)
        });

        return token;
    }

    /**
     * Validate username and password (private method, complexity = 4)
     * @private
     * @param {string} username - Username
     * @param {string} password - Password
     * @returns {boolean} True if valid
     */
    _validateCredentials(username, password) {
        if (password.length < 8) {  // +1
            return false;
        }

        if (!/[A-Z]/.test(password)) {  // +1
            return false;
        }

        if (!/[0-9]/.test(password)) {  // +1
            return false;
        }

        return true;
    }

    /**
     * Generate authentication token
     * @private
     * @param {string} username - Username
     * @returns {string} Generated token
     */
    _generateToken(username) {
        const crypto = require('crypto');
        const data = `${username}:${this.secretKey}:${Date.now()}`;
        return crypto.createHash('sha256').update(data).digest('hex');
    }
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        simpleGreeting,
        calculateGrade,
        complexValidator,
        doubleNumber,
        processNumbers,
        filterAndProcessUsers,
        User,
        AuthenticationManager
    };
}
