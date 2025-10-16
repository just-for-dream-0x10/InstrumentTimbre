# Makefile for Music AI Ecosystem
# Unified command management for code quality, formatting, and static analysis

.PHONY: help clean install lint format type-check test security audit fix-all check-all
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project directories
PROJECTS := MusicAITools InstrumentTimbre theory_net MusicEmotionAnalyzer AudioLayers
PYTHON_FILES := $(shell find $(PROJECTS) -name "*.py" -not -path "*/.*" -not -path "*/__pycache__/*" 2>/dev/null)
CONDA_ENV := myenv

help: ## Show this help message
	@echo "$(BLUE)Music AI Ecosystem - Unified Development Commands$(RESET)"
	@echo "=================================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(install|setup)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Quality Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(lint|format|type|check)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(test|security|audit)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utility Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v -E '(install|setup|lint|format|type|check|test|security|audit)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'

# Setup Commands
install-dev-tools: ## Install development tools for code quality (existing env)
	@echo "$(YELLOW)Installing development tools to existing $(CONDA_ENV) environment...$(RESET)"
	@echo "$(BLUE)Note: Using existing conda environment without modifications$(RESET)"
	@conda run -n $(CONDA_ENV) pip install -q --upgrade black isort flake8 mypy pylint bandit safety pytest pytest-cov 2>/dev/null || \
		echo "$(YELLOW)⚠️  Some tools may already be installed or incompatible$(RESET)"
	@echo "$(GREEN)✅ Development tools installation completed$(RESET)"

check-env: ## Check conda environment status
	@echo "$(YELLOW)Checking conda environment status...$(RESET)"
	@conda info --envs | grep $(CONDA_ENV) >/dev/null && \
		echo "$(GREEN)✅ Conda environment '$(CONDA_ENV)' exists$(RESET)" || \
		(echo "$(RED)❌ Conda environment '$(CONDA_ENV)' not found$(RESET)" && exit 1)
	@echo "$(BLUE)Python version:$(RESET)"
	@conda run -n $(CONDA_ENV) python --version
	@echo "$(BLUE)Key packages:$(RESET)"
	@conda run -n $(CONDA_ENV) pip list | grep -E "(torch|numpy|librosa|pytorch)" | head -5 || echo "  No key ML packages found"

# Code Formatting
format: ## Format all Python code with black and isort
	@echo "$(YELLOW)Formatting Python code...$(RESET)"
	@if [ -n "$(PYTHON_FILES)" ]; then \
		echo "Found $(shell echo $(PYTHON_FILES) | wc -w) Python files to format"; \
		conda run -n $(CONDA_ENV) black --line-length 88 --target-version py38 $(PYTHON_FILES); \
		conda run -n $(CONDA_ENV) isort --profile black $(PYTHON_FILES); \
		echo "$(GREEN)✅ Code formatting completed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  No Python files found$(RESET)"; \
	fi

format-check: ## Check if code formatting is needed (dry run)
	@echo "$(YELLOW)Checking code formatting...$(RESET)"
	@if [ -n "$(PYTHON_FILES)" ]; then \
		conda run -n $(CONDA_ENV) black --check --line-length 88 $(PYTHON_FILES) && \
		conda run -n $(CONDA_ENV) isort --check-only --profile black $(PYTHON_FILES) && \
		echo "$(GREEN)✅ Code formatting is correct$(RESET)" || \
		(echo "$(RED)❌ Code formatting issues found. Run 'make format' to fix$(RESET)" && exit 1); \
	fi

# Linting
lint: ## Run flake8 linting on all Python files
	@echo "$(YELLOW)Running flake8 linting...$(RESET)"
	@if [ -n "$(PYTHON_FILES)" ]; then \
		conda run -n $(CONDA_ENV) flake8 --max-line-length=88 --extend-ignore=E203,W503 $(PYTHON_FILES) && \
		echo "$(GREEN)✅ Linting passed$(RESET)" || \
		echo "$(RED)❌ Linting issues found$(RESET)"; \
	fi

lint-detailed: ## Run pylint for detailed code analysis
	@echo "$(YELLOW)Running detailed pylint analysis...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Analyzing $$project...$(RESET)"; \
			conda run -n $(CONDA_ENV) pylint $$project --disable=C0103,C0111,R0903 --output-format=colorized || true; \
		fi; \
	done

# Type Checking
type-check: ## Run mypy type checking
	@echo "$(YELLOW)Running mypy type checking...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Type checking $$project...$(RESET)"; \
			conda run -n $(CONDA_ENV) mypy $$project --ignore-missing-imports --follow-imports=silent || true; \
		fi; \
	done

# Security Analysis
security: ## Run security analysis with bandit
	@echo "$(YELLOW)Running security analysis...$(RESET)"
	@if [ -n "$(PYTHON_FILES)" ]; then \
		conda run -n $(CONDA_ENV) bandit -r $(PROJECTS) -f json -o security_report.json || true; \
		conda run -n $(CONDA_ENV) bandit -r $(PROJECTS) -ll || true; \
		echo "$(GREEN)✅ Security analysis completed$(RESET)"; \
	fi

audit: ## Audit dependencies for security vulnerabilities
	@echo "$(YELLOW)Auditing dependencies...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -f $$project/requirements.txt ]; then \
			echo "$(BLUE)Auditing $$project dependencies...$(RESET)"; \
			conda run -n $(CONDA_ENV) safety check -r $$project/requirements.txt || true; \
		fi; \
	done

# Testing
test: ## Run tests for all projects
	@echo "$(YELLOW)Running tests...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project/tests ]; then \
			echo "$(BLUE)Testing $$project...$(RESET)"; \
			cd $$project && conda run -n $(CONDA_ENV) pytest tests/ -v || true; \
			cd ..; \
		elif [ -f $$project/test*.py ]; then \
			echo "$(BLUE)Testing $$project...$(RESET)"; \
			cd $$project && conda run -n $(CONDA_ENV) python test*.py || true; \
			cd ..; \
		fi; \
	done

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project/tests ]; then \
			echo "$(BLUE)Testing $$project with coverage...$(RESET)"; \
			cd $$project && conda run -n $(CONDA_ENV) pytest tests/ --cov=. --cov-report=html --cov-report=term || true; \
			cd ..; \
		fi; \
	done

# Code Quality Metrics
complexity: ## Analyze code complexity
	@echo "$(YELLOW)Analyzing code complexity...$(RESET)"
	@conda run -n $(CONDA_ENV) pip install -q radon 2>/dev/null || echo "$(YELLOW)Installing radon...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Complexity analysis for $$project:$(RESET)"; \
			conda run -n $(CONDA_ENV) radon cc $$project -a -s --total-average || true; \
			echo ""; \
		fi; \
	done
	@echo "$(GREEN)✅ Complexity analysis completed$(RESET)"

dead-code: ## Find potentially dead/unused code
	@echo "$(YELLOW)Searching for potentially dead code...$(RESET)"
	@conda run -n $(CONDA_ENV) pip install -q vulture 2>/dev/null || echo "$(YELLOW)Installing vulture...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Dead code analysis for $$project:$(RESET)"; \
			conda run -n $(CONDA_ENV) vulture $$project --min-confidence 80 --sort-by-size || true; \
			echo ""; \
		fi; \
	done

duplicates: ## Find code duplicates
	@echo "$(YELLOW)Searching for code duplicates...$(RESET)"
	@conda run -n $(CONDA_ENV) pip install -q pyflakes 2>/dev/null || true
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Checking $$project for duplicates:$(RESET)"; \
			find $$project -name "*.py" -exec grep -l "def \|class " {} \; | \
			xargs -I {} sh -c 'echo "=== {} ==="; head -20 {} | grep -E "def |class "' | \
			sort | uniq -c | sort -nr | head -10 || true; \
			echo ""; \
		fi; \
	done

# Documentation
doc-check: ## Check documentation completeness
	@echo "$(YELLOW)Checking documentation...$(RESET)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)Checking $$project documentation...$(RESET)"; \
		if [ -f $$project/README.md ]; then \
			echo "  ✅ README.md exists"; \
		else \
			echo "  ❌ README.md missing"; \
		fi; \
		if [ -d $$project/docs ]; then \
			echo "  ✅ docs/ directory exists"; \
		else \
			echo "  ❌ docs/ directory missing"; \
		fi; \
	done

# Chinese Character Detection
check-chinese: ## Check for Chinese characters in code (violates rules.md)
	@echo "$(YELLOW)Checking for Chinese characters...$(RESET)"
	@found_chinese=false; \
	for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			chinese_files=$$(find $$project -name "*.py" -exec grep -l "[\u4e00-\u9fff]" {} \; 2>/dev/null || true); \
			if [ -n "$$chinese_files" ]; then \
				echo "$(RED)❌ Chinese characters found in $$project:$(RESET)"; \
				for file in $$chinese_files; do \
					echo "  - $$file"; \
					grep -n "[\u4e00-\u9fff]" $$file | head -3; \
				done; \
				found_chinese=true; \
			else \
				echo "$(GREEN)✅ No Chinese characters in $$project$(RESET)"; \
			fi; \
		fi; \
	done; \
	if [ "$$found_chinese" = true ]; then \
		echo "$(RED)❌ Chinese characters found! This violates ./aim/rules.md$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)✅ All code follows Chinese character policy$(RESET)"; \
	fi

# Print Statement Detection
check-print: ## Check for print statements (should use logger instead)
	@echo "$(YELLOW)Checking for print statements...$(RESET)"
	@found_print=false; \
	for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			print_files=$$(find $$project -name "*.py" -exec grep -l "print(" {} \; 2>/dev/null || true); \
			if [ -n "$$print_files" ]; then \
				echo "$(RED)❌ print() statements found in $$project:$(RESET)"; \
				for file in $$print_files; do \
					echo "  - $$file"; \
					grep -n "print(" $$file | head -3; \
				done; \
				found_print=true; \
			else \
				echo "$(GREEN)✅ No print statements in $$project$(RESET)"; \
			fi; \
		fi; \
	done; \
	if [ "$$found_print" = true ]; then \
		echo "$(YELLOW)⚠️  Print statements found! Consider using logger instead$(RESET)"; \
	else \
		echo "$(GREEN)✅ All code uses proper logging$(RESET)"; \
	fi

# Import Analysis
check-imports: ## Analyze import structure and detect issues
	@echo "$(YELLOW)Analyzing import structure...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Import analysis for $$project:$(RESET)"; \
			echo "  📦 Most common imports:"; \
			find $$project -name "*.py" -exec grep -h "^import\|^from.*import" {} \; 2>/dev/null | \
			sed 's/import //' | sed 's/from //' | cut -d' ' -f1 | cut -d'.' -f1 | \
			sort | uniq -c | sort -nr | head -10 || true; \
			echo "  🔍 Relative imports:"; \
			find $$project -name "*.py" -exec grep -n "from \." {} \; 2>/dev/null | head -5 || echo "    None found"; \
			echo "  ⚠️  Star imports:"; \
			find $$project -name "*.py" -exec grep -n "import \*" {} \; 2>/dev/null | head -5 || echo "    None found"; \
			echo ""; \
		fi; \
	done

dependencies: ## Analyze project dependencies
	@echo "$(YELLOW)Analyzing project dependencies...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -f $$project/requirements.txt ]; then \
			echo "$(BLUE)Dependencies for $$project:$(RESET)"; \
			echo "  📦 Total packages: $$(wc -l < $$project/requirements.txt)"; \
			echo "  🔒 Pinned versions: $$(grep -c "==" $$project/requirements.txt || echo 0)"; \
			echo "  🔓 Flexible versions: $$(grep -c ">=" $$project/requirements.txt || echo 0)"; \
			echo "  🚨 Heavy packages:"; \
			grep -E "(tensorflow|torch|opencv|scipy|sklearn|pandas)" $$project/requirements.txt || echo "    None detected"; \
			echo ""; \
		else \
			echo "$(YELLOW)No requirements.txt found for $$project$(RESET)"; \
		fi; \
	done

# Code Structure Analysis
architecture-check: ## Check architecture consistency and patterns
	@echo "$(YELLOW)Checking architecture consistency...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)Architecture analysis for $$project:$(RESET)"; \
			echo "  📁 Directory structure:"; \
			find $$project -type d -name "modules" -o -name "services" -o -name "core" -o -name "utils" | head -10 || echo "    Traditional structure"; \
			echo "  🏗️  Service classes:"; \
			find $$project -name "*.py" -exec grep -l "class.*Service" {} \; | wc -l | sed 's/^/    /' || echo "    0"; \
			echo "  📝 Configuration files:"; \
			find $$project -name "*.yaml" -o -name "*.json" -o -name "config.py" | head -5 || echo "    None found"; \
			echo "  🧪 Test files:"; \
			find $$project -name "test_*.py" -o -name "*_test.py" | wc -l | sed 's/^/    /' || echo "    0"; \
			echo ""; \
		fi; \
	done

todo-check: ## Find TODO, FIXME, HACK comments in code
	@echo "$(YELLOW)Searching for TODO/FIXME/HACK comments...$(RESET)"
	@for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			echo "$(BLUE)TODO analysis for $$project:$(RESET)"; \
			todos=$$(find $$project -name "*.py" -exec grep -n -E "(TODO|FIXME|HACK|XXX)" {} \; 2>/dev/null || true); \
			if [ -n "$$todos" ]; then \
				echo "$$todos" | head -10; \
			else \
				echo "  ✅ No TODO/FIXME/HACK comments found"; \
			fi; \
			echo ""; \
		fi; \
	done

# Combined Checks
check-all: ## Run all code quality checks
	@echo "$(BLUE)Running comprehensive code quality analysis...$(RESET)"
	@echo "=================================================="
	@$(MAKE) check-env
	@$(MAKE) check-chinese
	@$(MAKE) check-print
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security
	@$(MAKE) complexity
	@$(MAKE) dead-code
	@$(MAKE) check-imports
	@$(MAKE) dependencies
	@$(MAKE) architecture-check
	@$(MAKE) todo-check
	@echo "$(GREEN)✅ All quality checks completed$(RESET)"

quick-check: ## Run essential quality checks only
	@echo "$(BLUE)Running quick quality checks...$(RESET)"
	@echo "======================================="
	@$(MAKE) check-chinese
	@$(MAKE) check-print
	@$(MAKE) format-check
	@$(MAKE) lint
	@echo "$(GREEN)✅ Quick checks completed$(RESET)"

fix-all: ## Automatically fix what can be fixed
	@echo "$(BLUE)Auto-fixing code issues...$(RESET)"
	@echo "=============================="
	@$(MAKE) format
	@echo "$(GREEN)✅ Auto-fixes completed. Run 'make check-all' to verify$(RESET)"

# Utility Commands
clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up generated files...$(RESET)"
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f security_report.json 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

stats: ## Show project statistics
	@echo "$(BLUE)Music AI Ecosystem Statistics$(RESET)"
	@echo "================================"
	@total_py_files=0; \
	total_lines=0; \
	for project in $(PROJECTS); do \
		if [ -d $$project ]; then \
			py_files=$$(find $$project -name "*.py" | wc -l); \
			lines=$$(find $$project -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $$1}' || echo 0); \
			echo "$(YELLOW)$$project:$(RESET)"; \
			echo "  📄 Python files: $$py_files"; \
			echo "  📝 Lines of code: $$lines"; \
			total_py_files=$$((total_py_files + py_files)); \
			total_lines=$$((total_lines + lines)); \
		fi; \
	done; \
	echo "$(GREEN)Total:$(RESET)"; \
	echo "  📄 Python files: $$total_py_files"; \
	echo "  📝 Lines of code: $$total_lines"

# Create quality report
quality-report: ## Generate comprehensive quality report
	@echo "$(BLUE)Generating Quality Report...$(RESET)"
	@echo "# Music AI Ecosystem Quality Report" > quality_report.md
	@echo "Generated on: $$(date)" >> quality_report.md
	@echo "" >> quality_report.md
	@echo "## Project Statistics" >> quality_report.md
	@$(MAKE) stats >> quality_report.md 2>&1
	@echo "" >> quality_report.md
	@echo "## Code Quality Checks" >> quality_report.md
	@echo "### Chinese Character Check" >> quality_report.md
	@$(MAKE) check-chinese >> quality_report.md 2>&1 || true
	@echo "" >> quality_report.md
	@echo "### Print Statement Check" >> quality_report.md
	@$(MAKE) check-print >> quality_report.md 2>&1 || true
	@echo "" >> quality_report.md
	@echo "### Import Analysis" >> quality_report.md
	@$(MAKE) check-imports >> quality_report.md 2>&1 || true
	@echo "$(GREEN)✅ Quality report generated: quality_report.md$(RESET)"