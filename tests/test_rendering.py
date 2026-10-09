from factory_frontier import FactoryFrontier


def test_ascii_render_dimensions() -> None:
    env = FactoryFrontier("intro", render_mode="ansi")
    env.reset(seed=0)
    rendered = env.render()
    assert len(rendered.splitlines()) == 5
    assert "F" in rendered
