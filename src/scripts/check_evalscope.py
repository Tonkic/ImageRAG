
try:
    import evalscope
    print(f"Evalscope version: {evalscope.__version__}")

    try:
        from evalscope.metrics import CLIPScoreMetric, ImageRewardMetric
        print("Import from evalscope.metrics: SUCCESS")
    except ImportError as e:
        print(f"Import from evalscope.metrics: FAILED ({e})")

    try:
        from evalscope.metrics.vl import CLIPScoreMetric, ImageRewardMetric
        print("Import from evalscope.metrics.vl: SUCCESS")
    except ImportError as e:
        print(f"Import from evalscope.metrics.vl: FAILED ({e})")

    import inspect
    print("\nContents of evalscope.metrics:")
    try:
        if hasattr(evalscope.metrics, '__all__'):
             print(evalscope.metrics.__all__)
        else:
             print(dir(evalscope.metrics))
    except Exception as e:
        print(e)

except ImportError:
    print("Evalscope not installed")
