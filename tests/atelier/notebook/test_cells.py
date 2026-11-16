"""`cells.py` is the notebook's grammar: nbformat-4 dicts, the `@@NAME@@`
template language and the static check every built or patched notebook
must pass. These tests pin the failure modes a reader would otherwise meet in
the kernel: an unfilled placeholder, a cell that does not parse, a missing
section, a backend function whose source cannot be read."""

from __future__ import annotations

import pytest

from sakura.atelier.notebook import cells


def test_fill_replaces_every_placeholder() -> None:
    assert cells.fill("a=@@A@@; b=@@B@@", A="1", B="2") == "a=1; b=2"


def test_fill_refuses_an_unfilled_placeholder() -> None:
    with pytest.raises(cells.TemplateError, match="@@B@@"):
        cells.fill("@@A@@ @@B@@", A="1")


def test_fill_does_not_rescan_substituted_values() -> None:
    # A value that itself looks like a placeholder is data, not a template.
    assert cells.fill("x=@@A@@", A="@@B@@") == "x=@@B@@"


def test_source_of_reads_a_function() -> None:
    def f() -> int:
        return 1

    src, note = cells.source_of(f)
    assert "def f()" in src and note is None


def test_source_of_an_unreadable_object_cites_it_instead() -> None:
    src, note = cells.source_of(len)
    assert src == ""
    assert note is not None and "builtins.len" in note


def test_check_notebook_flags_missing_heading_bad_syntax_and_placeholder() -> None:
    nb = cells.notebook([cells.markdown("## Setup"), cells.code("x = (")], title="t")
    problems = cells.check_notebook(nb)
    assert any("missing section: '## Data'" in p for p in problems)
    assert any("syntax error" in p for p in problems)
    nb2 = cells.notebook(
        [cells.markdown(h) for h in cells.SECTION_HEADINGS] + [cells.code("@@X@@")],
        title="t",
    )
    assert any("placeholder" in p for p in cells.check_notebook(nb2))


def test_check_notebook_tolerates_ipython_magics() -> None:
    body = [cells.markdown(h) for h in cells.SECTION_HEADINGS] + [
        cells.code("%matplotlib inline\n!ls\nprint(1)")
    ]
    assert cells.check_notebook(cells.notebook(body, title="ok")) == []


def test_check_notebook_rejects_a_wrong_nbformat() -> None:
    nb = cells.notebook([cells.markdown(h) for h in cells.SECTION_HEADINGS], title="t")
    nb["nbformat"] = 3
    assert any("nbformat must be 4" in p for p in cells.check_notebook(nb))


def test_a_complete_notebook_passes_the_check() -> None:
    body = [cells.markdown(h) for h in cells.SECTION_HEADINGS] + [
        cells.code("print(1)"),
        cells.markdown(cells.REPAIRS_HEADING),
    ]
    assert cells.check_notebook(cells.notebook(body, title="ok")) == []


def test_source_text_joins_list_sources() -> None:
    assert cells.source_text({"source": ["a\n", "b"]}) == "a\nb"
    assert cells.source_text({"source": "c"}) == "c"
    assert cells.source_text({}) == ""


def test_notebook_carries_kernelspec_title_and_extra_metadata() -> None:
    nb = cells.notebook([], title="run", metadata={"extra": 1})
    assert nb["nbformat"] == 4 and nb["cells"] == []
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["metadata"]["atelier"]["title"] == "run"
    assert nb["metadata"]["extra"] == 1


def test_code_and_markdown_cells_have_nbformat_shape() -> None:
    md = cells.markdown("## Setup", tag="x")
    assert md == {"cell_type": "markdown", "metadata": {"tag": "x"}, "source": "## Setup"}
    out = cells.stream_output("hi", name="stderr")
    c = cells.code("print(1)", outputs=[out], atelier={"section": "Setup"})
    assert c["cell_type"] == "code" and c["execution_count"] is None
    assert c["outputs"] == [{"output_type": "stream", "name": "stderr", "text": "hi"}]
    assert c["metadata"] == {"atelier": {"section": "Setup"}}


def test_html_output_is_display_data_with_plain_fallback() -> None:
    out = cells.html_output("<b>x</b>", text="x")
    assert out["output_type"] == "display_data"
    assert out["data"] == {"text/html": "<b>x</b>", "text/plain": "x"}


def test_png_output_is_base64_display_data() -> None:
    out = cells.png_output(b"\x89PNG", text="curve")
    assert out["output_type"] == "display_data" and out["data"]["image/png"] == "iVBORw=="
    assert out["data"]["text/plain"] == "curve"


def test_section_headings_are_the_contract_order() -> None:
    assert cells.SECTION_HEADINGS == (
        "## Setup", "## Data", "## Model", "## Training",
        "## Evaluation", "## Export", "## Package",
    )
    assert cells.REPAIRS_HEADING == "## Repairs"
