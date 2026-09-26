"""CyclopsDiary: a shared visual memory for smart glasses and phone cameras.

Every camera in a workspace is read by a world model (nvidia/Cosmos3-Edge,
the recording read as one continuous video with the model's memory carried)
and every second of it lands in MongoDB as the model's own output-head row.
A person's object tracker is private; it finds the object again in anyone's
footage by example, and its last known location is read off the sightings.

    config     settings from the environment and .env
    atlas      the MongoDB connection, collections and indexes
    workspace  people and their cameras
    tower      the world model, wired (side stack, frame size, rotation)
    ingest     footage -> steps in MongoDB
    memory     the workspace's steps read back and searched by example
    tracker    private trackers, sightings, the last known location
    objects    a person's objects: found by the model's own words for them and their examples; trail, last seen
    query      one example in (forward, or reversed: the example undone), events out
    stream     one worker: every camera's inbox ingested, queued queries answered
    agentmemory  the agent's conversations, notes and starting context, in MongoDB
    tools      the agent's tools (queries, timeline, memory) as plain functions
    mcp_server   those tools over MCP (`bin/cyclopsdiary mcp`)
    cli        the `bin/cyclopsdiary` command line
"""
