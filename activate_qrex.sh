#!/bin/bash
echo "🚀 Activating QREX-FL environment..."

# Activate virtual environment
if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Set environment variables
export QREX_FL_ROOT=$(pwd)
export PYTHONPATH="${QREX_FL_ROOT}/src:${PYTHONPATH}"
export QREX_FL_ENV=development

echo "✅ QREX-FL environment activated"
echo "📍 Project root: $QREX_FL_ROOT"
echo ""
echo "🔧 Available commands:"
echo "  python examples/demo.py              - Run compliance demo"
echo "  python examples/federated_training.py - Start federated training"
echo "  python src/api/main.py               - Start API server"
echo "  pytest tests/                        - Run test suite"
echo "  python scripts/download_datasets.py  - Download datasets"
echo ""
echo "🌐 After starting API server:"
echo "  http://localhost:8000                - API root"
echo "  http://localhost:8000/docs           - Interactive API docs"
echo "  http://localhost:8000/health         - Health check"
echo ""
echo "Ready for QREX-FL development! 🚀"
