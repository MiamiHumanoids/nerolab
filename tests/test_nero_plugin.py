import importlib


def test_package_is_importable():
    mod = importlib.import_module("lerobot_robot_nero")
    assert hasattr(mod, "NeroConfig")
    assert hasattr(mod, "Nero")
    assert hasattr(mod, "NeroRobot")


def test_robot_config_registration():
    from lerobot.robots import RobotConfig
    from lerobot_robot_nero import NeroConfig

    cfg = NeroConfig(
        id="test-arm",
        calibration_dir=None,
        can_interface="socketcan",
        can_channel="can0",
        bitrate=1_000_000,
    )

    assert cfg.type == "nero"
    assert RobotConfig.get_choice_name(type(cfg)) == "nero"


def test_cli_entry_point_exists():
    import importlib.metadata as md

    eps = md.entry_points(group="console_scripts")
    names = {ep.name for ep in eps}
    assert "lerobot-nero" in names
