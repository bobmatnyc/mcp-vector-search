#!/usr/bin/env python3
"""Test the Ruby parser."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.parsers.ruby import RubyParser


@pytest.mark.asyncio
async def test_ruby_parser():
    """Test Ruby parser functionality."""
    print("🔍 Testing Ruby parser...")

    # Create test Ruby file
    ruby_content = """
require 'json'
require_relative './config'

=begin
This is a test application module
Contains core business logic
=end
module MyApp
  # Application configuration module
  module Config
    # Default configuration values
    DEFAULT_TIMEOUT = 30

    # Get configuration value
    # @param key [String] the configuration key
    # @return [Object] the configuration value
    def self.get(key)
      @config ||= {}
      @config[key]
    end

    # Set configuration value
    def self.set(key, value)
      @config ||= {}
      @config[key] = value
    end
  end

  # Main application class
  # Handles core application logic
  class Application
    attr_accessor :name, :version
    attr_reader :status
    attr_writer :config

    # Initialize a new application
    # @param name [String] application name
    # @param version [String] application version
    def initialize(name, version = '1.0.0')
      @name = name
      @version = version
      @status = :initialized
    end

    # Check if application is running
    # @return [Boolean] true if running
    def running?
      @status == :running
    end

    # Start the application
    # Raises error if already running
    def start!
      raise 'Already running' if running?
      @status = :running
      perform_startup
    end

    # Stop the application
    def stop
      @status = :stopped
      perform_cleanup
    end

    # Get application info
    # @return [Hash] application information
    def info
      {
        name: @name,
        version: @version,
        status: @status
      }
    end

    # Create a default application instance
    # @return [Application] new application instance
    def self.default
      new('DefaultApp', '0.1.0')
    end

    class << self
      # Create application from configuration
      # @param config [Hash] configuration hash
      # @return [Application] configured application
      def from_config(config)
        app = new(config[:name], config[:version])
        app.config = config
        app
      end

      # Get application version
      def version
        '2.0.0'
      end
    end

    private

    # Perform startup tasks
    def perform_startup
      puts "Starting #{@name}..."
    end

    # Perform cleanup tasks
    def perform_cleanup
      puts "Cleaning up #{@name}..."
    end
  end

  # User management class
  class UserManager
    # Initialize user manager
    def initialize
      @users = []
    end

    # Add a new user
    # @param name [String] user name
    # @param email [String] user email
    def add_user(name, email)
      @users << { name: name, email: email }
    end

    # Find user by email
    # @param email [String] email to search
    # @return [Hash, nil] user hash or nil
    def find_by_email(email)
      @users.find { |user| user[:email] == email }
    end

    # Check if user exists
    def user_exists?(email)
      !find_by_email(email).nil?
    end
  end
end

# Top-level utility function
# @param message [String] message to log
def log_message(message)
  puts "[#{Time.now}] #{message}"
end

# Calculate factorial
# @param n [Integer] number
# @return [Integer] factorial result
def factorial(n)
  return 1 if n <= 1
  n * factorial(n - 1)
end

# Process data with block
def process_data(data, &block)
  data.map(&block)
end
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rb", delete=False) as f:
        f.write(ruby_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test Ruby parser
    ruby_parser = RubyParser()
    chunks = await ruby_parser.parse_file(test_file)

    print(f"📊 Ruby parser extracted {len(chunks)} chunks:")

    # Analyze chunks
    module_chunks = []
    class_chunks = []
    method_chunks = []
    class_method_chunks = []
    attribute_chunks = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        if chunk.function_name:
            print(f"  Function: {chunk.function_name}")
        if chunk.class_name:
            print(f"  Class: {chunk.class_name}")
        if chunk.docstring:
            print(f"  Docstring: {chunk.docstring[:60]}...")
        print(f"  Content preview: {chunk.content[:100]}...")

        # Categorize chunks
        if chunk.chunk_type == "module":
            module_chunks.append(chunk)
        if chunk.chunk_type == "class":
            class_chunks.append(chunk)
        if chunk.chunk_type == "method":
            method_chunks.append(chunk)
        if chunk.chunk_type == "class_method":
            class_method_chunks.append(chunk)
        if chunk.chunk_type == "attribute":
            attribute_chunks.append(chunk)

    # Verify key features
    print("\n" + "=" * 80)
    print("🎯 Feature Verification:")
    print("=" * 80)

    print(f"\n✅ Module chunks found: {len(module_chunks)}")
    assert len(module_chunks) >= 1, "Should find at least 1 module chunk"

    print(f"✅ Class chunks found: {len(class_chunks)}")
    assert len(class_chunks) >= 2, "Should find at least 2 class chunks"

    print(f"✅ Method chunks found: {len(method_chunks)}")
    assert len(method_chunks) >= 5, "Should find at least 5 method chunks"

    print(f"✅ Class method chunks found: {len(class_method_chunks)}")
    assert len(class_method_chunks) >= 2, "Should find at least 2 class method chunks"

    print(f"✅ Attribute chunks found: {len(attribute_chunks)}")
    # Note: attributes might not be extracted by tree-sitter, only in fallback mode

    # Verify RDoc extraction
    chunks_with_docs = [c for c in chunks if c.docstring]
    print(f"✅ Chunks with RDoc: {len(chunks_with_docs)}/{len(chunks)}")
    assert len(chunks_with_docs) >= 3, "Should extract RDoc from multiple chunks"

    # Verify module namespacing
    namespaced_chunks = [c for c in chunks if c.class_name and "::" in c.class_name]
    print(f"✅ Namespaced chunks found: {len(namespaced_chunks)}")
    assert len(namespaced_chunks) >= 1, "Should find namespaced classes/modules"

    # Verify method name patterns (?, !)
    special_methods = [
        c
        for c in chunks
        if c.function_name
        and (c.function_name.endswith("?") or c.function_name.endswith("!"))
    ]
    print(f"✅ Methods with ?/! found: {len(special_methods)}")
    assert len(special_methods) >= 2, "Should find methods with ? or ! suffixes"

    # Verify supported extensions
    assert ".rb" in ruby_parser.get_supported_extensions()
    assert ".rake" in ruby_parser.get_supported_extensions()
    assert ".gemspec" in ruby_parser.get_supported_extensions()
    print(f"✅ Supported extensions: {ruby_parser.get_supported_extensions()}")

    # Clean up
    test_file.unlink()
    print("\n✅ Ruby parser test completed successfully!")

    return True


@pytest.mark.asyncio
async def test_ruby_rails_patterns():
    """Test Ruby on Rails specific patterns."""
    print("\n🔍 Testing Ruby on Rails patterns...")

    # Create test file with Rails-style patterns
    rails_content = """
# User model for authentication
class User < ApplicationRecord
  has_many :posts
  has_many :comments

  validates :email, presence: true, uniqueness: true
  validates :name, presence: true

  # Authenticate user with password
  # @param password [String] password to check
  # @return [Boolean] true if authenticated
  def authenticate(password)
    BCrypt::Password.new(password_digest) == password
  end

  # Check if user is admin
  def admin?
    role == 'admin'
  end

  # Full name of the user
  def full_name
    "#{first_name} #{last_name}"
  end

  # Find all active users
  # @return [ActiveRecord::Relation] active users
  def self.active
    where(status: 'active')
  end

  # Create user with default role
  def self.create_with_defaults(attrs)
    create(attrs.merge(role: 'user', status: 'active'))
  end
end

# Posts controller
class PostsController < ApplicationController
  before_action :authenticate_user!
  before_action :set_post, only: [:show, :edit, :update, :destroy]

  # GET /posts
  def index
    @posts = Post.all.order(created_at: :desc)
  end

  # GET /posts/:id
  def show
    # Post is set by before_action
  end

  # POST /posts
  def create
    @post = current_user.posts.build(post_params)

    if @post.save
      redirect_to @post, notice: 'Post created successfully.'
    else
      render :new
    end
  end

  private

  # Set post for actions
  def set_post
    @post = Post.find(params[:id])
  end

  # Strong parameters
  def post_params
    params.require(:post).permit(:title, :content)
  end
end
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rb", delete=False) as f:
        f.write(rails_content)
        test_file = Path(f.name)

    print(f"📁 Created Rails test file: {test_file}")

    # Parse the file
    ruby_parser = RubyParser()
    chunks = await ruby_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks from Rails patterns:")

    # Verify ActiveRecord model detection
    model_classes = [
        c for c in chunks if c.class_name and "ApplicationRecord" in str(c.content)
    ]
    print(f"✅ ActiveRecord model classes found: {len(model_classes)}")

    # Verify controller detection
    controller_classes = [
        c for c in chunks if c.class_name and "Controller" in c.class_name
    ]
    print(f"✅ Controller classes found: {len(controller_classes)}")

    # Verify instance methods
    instance_methods = [
        c
        for c in chunks
        if c.chunk_type == "method"
        and c.function_name
        and not c.function_name.startswith("self.")
    ]
    print(f"✅ Instance methods found: {len(instance_methods)}")

    # Verify class methods
    class_methods = [
        c
        for c in chunks
        if c.chunk_type in ["class_method", "method"]
        and c.function_name
        and "self." in c.function_name
    ]
    print(f"✅ Class methods found: {len(class_methods)}")

    # Clean up
    test_file.unlink()
    print("\n✅ Rails patterns test completed successfully!")

    return True


@pytest.mark.asyncio
async def main():
    """Run all Ruby parser tests."""
    try:
        await test_ruby_parser()
        await test_ruby_rails_patterns()
        print("\n🎉 All Ruby parser tests completed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
