"""
The "infrastructure" folder: everything that talks to the outside world.

    provider.py    calls the AI companies' APIs, through LiteLLM
    otel.py        sends the measurements to OpenTelemetry
    persistence/   reads and writes the ClickHouse database

Why keep all of this in one place? Because these are the slow, unreliable
parts - the network might be down, the API key might be wrong, the database
might not be running. Corralling them behind one folder means a test can swap
any of them for a pretend version and check the real logic without ever
touching the network.
"""
