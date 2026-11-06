# The Demo world must hit 30 fps at 720p on the floor GPU, aiming for 1080p at 60 fps

Performance is the top priority, and every rendering feature needs a scalable lower-cost mode so the full feature set runs on the floor GPU (a Ryzen 7 3700U Vega-class iGPU, see ADR 0005). The committed target is the Demo world at 30 fps, 720p, low settings on that machine. The ambition, pursued through design and optimization and not promised, is 1080p at 60 fps. A measurable target on real hardware makes performance regressions detectable and prevents scalability from being deferred to the end.
