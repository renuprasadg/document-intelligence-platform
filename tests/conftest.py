"""
pytest fixtures for GuardianRAG Week 2 tests.
 
Provides:
  - sample_text: A multi-paragraph clean text fixture
  - multi_page_text: Simulated multi-page PDF extraction
  - dirty_text: Text with normalization artifacts
  - tmp_pdf: A minimal real PDF created with PyMuPDF (for extraction tests)
"""
from pathlib import Path
import pytest
 
 
CLEAN_SAMPLE = """
Artificial Intelligence in Healthcare
 
Artificial intelligence (AI) is transforming healthcare delivery across the globe.
Machine learning algorithms can now detect cancers from medical imaging with
accuracy rivalling expert radiologists. Natural language processing extracts
structured data from unstructured clinical notes at scale.
 
Clinical Decision Support
 
AI-powered clinical decision support systems alert physicians to potential drug
interactions, flag deteriorating patients in ICUs, and suggest evidence-based
treatment pathways. These systems learn continuously from patient outcome data.
 
Challenges and Limitations
 
Despite rapid advances, AI healthcare systems face challenges around data
privacy, algorithmic bias, and regulatory approval. Explainability remains
a central concern: clinicians need to understand why a model made a prediction
before acting on it.
""".strip()
 
 
MULTI_PAGE = "\x0c".join([
    "COMPANY ANNUAL REPORT 2024\nRevenue grew 15% year-on-year to $2.4 billion.\n1",
    "COMPANY ANNUAL REPORT 2024\nOperating expenses increased by 8% to $1.8 billion.\n2",
    "COMPANY ANNUAL REPORT 2024\nNet income was $600 million, up from $520 million.\n3",
    "COMPANY ANNUAL REPORT 2024\nCash and equivalents stood at $1.1 billion at year end.\n4",
    "COMPANY ANNUAL REPORT 2024\nThe board approved a 10% increase in the dividend.\n5",
])
 
 
DIRTY_TEXT = (
    "Ef\ufb03cient data pro\u00adcessing is critical for mo\u00addern sys\u00adtems. "
    "The results\u2014both quantitative and qualitative\u2014were signi\ufb01cant. "
    "\u201cAll teams\u201d agreed on the \u2018final\u2019 approach."
)
 
 
@pytest.fixture
def sample_text() -> str:
    """Clean multi-paragraph text fixture."""
    return CLEAN_SAMPLE
 
 
@pytest.fixture
def multi_page_text() -> str:
    """Simulated multi-page PDF extraction with repeating header/footer."""
    return MULTI_PAGE
 
 
@pytest.fixture
def dirty_text() -> str:
    """Text with ligatures, soft-hyphens, and smart typography."""
    return DIRTY_TEXT
 
 
@pytest.fixture
def tmp_pdf(tmp_path) -> Path:
    """
    Create a minimal real PDF using PyMuPDF for extraction tests.
 
    Returns path to the created PDF file.
    """
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed - skipping PDF fixture")
 
    pdf_path = tmp_path / "test_document.pdf"
    doc = fitz.open()
 
    for i in range(3):
        page = doc.new_page()
        text = (
            f"Page {i + 1} of the test document.\n"
            f"This page contains sample content for testing extraction.\n"
            f"The GuardianRAG system processes documents automatically."
        )
        page.insert_text((72, 72), text, fontsize=12)
 
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
 
 
# ─── STATIC FIXTURE PDFs (created once, committed to repo) ─────────────
# Run this once to generate the fixture PDFs in tests/fixtures/sample_pdfs/
#
# python -c "from tests.conftest import create_fixture_pdfs; create_fixture_pdfs()"
 
 
def create_fixture_pdfs() -> None:
    """
    Generate static fixture PDFs committed to the repo.
    Call once during project setup.
    Output: tests/fixtures/sample_pdfs/policy_sample.pdf
            tests/fixtures/sample_pdfs/claims_guide.pdf
    """
    import fitz
    fixtures_dir = Path("tests/fixtures/sample_pdfs")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
 
    # policy_sample.pdf - simulates a multi-page insurance policy
    policy_path = fixtures_dir / "policy_sample.pdf"
    if not policy_path.exists():
        doc = fitz.open()
        pages = [
            "GuardianRAG Policy Sample\n\nSection 1: Coverage Overview\n\n"
            "This policy provides comprehensive coverage for digital assets.\n"
            "Coverage includes data loss, unauthorized access, and system downtime.",
            "GuardianRAG Policy Sample\n\nSection 2: Exclusions\n\n"
            "The following are excluded from coverage: acts of war, natural disasters,\n"
            "and negligent security practices by the insured party.",
            "GuardianRAG Policy Sample\n\nSection 3: Claims Process\n\n"
            "To file a claim, contact our claims department within 30 days.\n"
            "Provide documentation of the incident and estimated losses.",
        ]
        for text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=12)
        doc.save(str(policy_path))
        doc.close()
        print(f"Created: {policy_path}")
 
    # claims_guide.pdf - simulates a claims processing guide
    claims_path = fixtures_dir / "claims_guide.pdf"
    if not claims_path.exists():
        doc = fitz.open()
        pages = [
            "Claims Processing Guide\n\nChapter 1: Initial Assessment\n\n"
            "All claims must be assessed within 48 hours of receipt.\n"
            "The initial assessment determines claim validity and priority.",
            "Claims Processing Guide\n\nChapter 2: Documentation Requirements\n\n"
            "Claimants must provide: proof of loss, incident timeline,\n"
            "and supporting technical evidence where applicable.",
        ]
        for text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=12)
        doc.save(str(claims_path))
        doc.close()
        print(f"Created: {claims_path}")
