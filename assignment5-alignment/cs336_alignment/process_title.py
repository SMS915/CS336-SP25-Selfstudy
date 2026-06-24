def set_python_process_title() -> None:
    try:
        import setproctitle

        setproctitle.setproctitle("python")
    except Exception:
        # Best-effort only: training should still run even if the optional
        # dependency is unavailable.
        return
