# --- Paper build targets ---
PAPER = paper
TEMPLATE = joss-template.latex

.PHONY: all install pdf clean test examples suc mlp lenet cnn bnn

# Default target: install deps, build pdf, run tests + examples
all: install pdf test examples

# --- Installation ---
install:
	@echo "📦 Installing package with dev extras..."
	pip install -e '.[dev]'

# --- Paper build ---
pdf: $(PAPER).md $(PAPER).bib $(TEMPLATE)
	pandoc $(PAPER).md \
		--from=markdown \
		--pdf-engine=xelatex \
		--citeproc \
		--bibliography=$(PAPER).bib \
		--template=$(TEMPLATE) \
		--syntax-highlighting=none \
		-o $(PAPER).pdf

clean:
	rm -f $(PAPER).pdf

# --- Testing ---
test: install
	pytest -q --cov=freegrad --cov-report=term-missing

# --- Examples ---
examples: install suc mlp lenet cnn bnn
	@echo "\n✅ All examples completed."

suc:
	@echo "Running SUC (Logistic vs Constant)..."
	@python examples/suc_logistic_vs_constant.py

mlp:
	@echo "\nRunning MLP on DIGITS..."
	@python examples/mlp_digits_constant_vs_tied.py

lenet:
	@echo "\nRunning LeNet on MNIST with Rectangular gradient..."
	@python examples/lenet_mnist_rectangular.py

cnn:
	@echo "\nRunning CNN with Gradient Jamming..."
	@python examples/cnn_gradient_jamming.py

bnn:
	@echo "\nRunning BNN with Step activation..."
	@python examples/bnn_step_activation.py
