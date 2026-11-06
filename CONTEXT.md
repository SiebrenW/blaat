# Engine

A data-oriented 3D game engine, with an in-engine editor, for large open-world action RPGs.

## Language

**Engine**:
The native runtime and editor that owns all world data and rendering, and loads the Game module.
_Avoid_: Framework, runtime (alone)

**Game module**:
A compiled, hot-reloadable unit of game logic written against the SDK. It contains only logic; all state lives in the Engine. The Engine loads several at once, and each declares which others it depends on.
_Avoid_: Script, plugin, mod

**Expansion**:
Additional content that can be installed or not, made of one or more Game modules plus Assets and authored content. A base game must run without it.
_Avoid_: DLC, add-on, patch

**SDK**:
The ergonomic library a Game module is written against. It is the only supported way for a Game module to talk to the Engine.
_Avoid_: API, bindings

**Component**:
A unit of data attached to an entity. Engine and Game module Components are registered with the Engine by ID, so both sides agree on what they mean.
_Avoid_: Property, attribute, field

**Component schema**:
The declared shape of a Component: its fields, types, and defaults. A change to a Component schema is migrated in place when fields are added or removed, and refused when a field's type changes or it is renamed.
_Avoid_: Layout, struct definition, format

**Entity**:
An identity that Components are attached to; it carries no data itself. Each Entity is owned by exactly one Cell, Interior, or the World.
_Avoid_: Object, actor, node, game object

**Persistent entity**:
An Entity the author explicitly flags as owned by the World and not by any Cell, so it outlives Cell streaming. Entities are Cell-owned by default. While its Cell is unloaded it is only tracked abstractly (position and schedule), without full simulation.
_Avoid_: Global entity, singleton, always-loaded entity

**System**:
A Game module function that reads and writes declared Components. The Engine decides when it runs and which Systems may run in parallel.
_Avoid_: Script, behavior, job, tick function

**World**:
The whole playable space of a game, composed of Cells and Interiors.
_Avoid_: Level, map, scene

**Cell**:
A fixed-size square of the World's outdoor grid; the single unit of streaming, saving, and editor ownership.
_Avoid_: Chunk, tile, sector, region

**Cell origin**:
The point in World coordinates that a Cell's stored positions are relative to. Items in a Cell store f32 positions relative to it.
_Avoid_: Cell anchor, local origin, chunk offset

**Terrain**:
The ground surface of a Cell, stored as a heightfield. Cliffs, overhangs, and cave entrances are not Terrain; they are mesh Assets placed on top of it.
_Avoid_: Landscape, ground mesh, voxel terrain

**Static instance**:
A lightweight placed item with a mesh Asset and a transform, such as a tree, rock, or fence post, stored as packed data in a Cell without being an Entity. It has a stable ID, so Overrides can target it, and can be promoted to an Entity when it needs Components, such as a choppable tree or a pushable crate.
_Avoid_: Foliage instance, prop, scatter item, tree entity

**Interior**:
A hand-authored enclosed space, anchored at a placement inside its house's footprint in the World's coordinates, with its own baked lighting. It is owned separately from the Cell grid and streamed in Interior tiers.
_Avoid_: Indoor Cell, room, building

**Interior tier**:
How much of an Interior is loaded, set by proximity and accessibility: *distant* (minimal interior rendering), *near* (reduced), and *accessible* (a doorway the player or an NPC can use is in range, so the Interior's Entities are loaded).
_Avoid_: Interior LOD, interior detail level

**Portal**:
A doorway or opening connecting an Interior to the World or to another Interior, in the same continuous space by default.
_Avoid_: Door, entrance, teleporter

**Remote portal**:
An opt-in Portal that connects to a different location instead of the adjacent space, used when an Interior must differ from its footprint, such as a larger quest space or a swapped layout.
_Avoid_: Teleport, warp, loading door

**Brush**:
A solid shape placed in the Editor to block out the geometry of houses and Interiors.
_Avoid_: Blockout, primitive, CSG shape

**Lighting state**:
One of a small, per-Interior set of baked lighting conditions (such as dawn, day, dusk, night), blended according to the World's time of day. Individual light sources can be baked as toggleable layers.
_Avoid_: Lightmap set, lighting variant, time-of-day bake

**Procedural rule**:
Authored source data in the editor that generates World content, such as terrain, foliage scatter, or roads. It is the source of truth for the content it generates.
_Avoid_: Generator, brush, script

**Override**:
A hand edit attached to one specific item a Procedural rule generated, so re-running the rule does not discard it. Items have stable identities across re-runs.
_Avoid_: Tweak, manual edit, patch

**Orphaned override**:
An Override whose target item no longer exists after its Procedural rule changed. It is surfaced to the author, never silently dropped.
_Avoid_: Stale override, dangling edit

**Bake**:
The offline step that turns Procedural rules, Overrides, lighting, and source Assets in open formats into the concrete, Engine-native data the shipped game loads. The shipped game runs no Procedural rules and imports no source files.
_Avoid_: Build, compile, generate, import

**Asset**:
A reusable piece of content such as a mesh, texture, material, or sound, referenced everywhere by a stable ID and never by path, so it survives renames and moves.
_Avoid_: Resource, file, content item

**Editor**:
The dev-only authoring application built into the Engine, used by humans and, through the MCP hook, by LLMs. It is not shipped with games.
_Avoid_: Tool, level editor, IDE

**Game UI**:
The retained-mode UI system available to Game modules for menus, HUD, and dialogue. It is separate from the Editor's own interface.
_Avoid_: HUD (alone), GUI, widget system

**Command**:
A named, validated, undoable editor action. Every change a human or an LLM makes to the authored World goes through a Command.
_Avoid_: Action, operation, tool call

**Live command**:
A non-undoable action against a running game session, such as spawning an Entity or teleporting the player. It never changes the authored World.
_Avoid_: Debug command, cheat, console command

**Save game**:
The persisted state of a play session, stored as differences from the baked World, plus Persistent entities and Game module Components in full. It is forward-compatible with game updates but not with downgrades. Data belonging to an uninstalled Game module is kept dormant and restored if the module returns.
_Avoid_: Snapshot, savefile, checkpoint

**Orphaned delta**:
A part of a Save game whose target Entity no longer exists after a game update. It is reported and dropped, never fatal.
_Avoid_: Stale save entry, broken reference

**Route graph**:
A small, always-loaded graph of travel edges between Cells, Interiors, and Portals, produced by the Bake. It plans long routes and gives travel times for Persistent entities while their Cells are unloaded.
_Avoid_: World navmesh, waypoint graph, nav graph

**Nav gate**:
A named, authored toggle on a Route graph edge, set by the Game module (for example when a quest advances). Both the Route graph and the local navmesh obey it, and it is persisted in the Save game. Obstructions that arise at runtime, like a barrel, are not Nav gates and do not affect the Route graph.
_Avoid_: Navigation hint, nav flag, path blocker

**Goal**:
Something an NPC wants to do, with conditions, a destination, and a priority. The Engine picks the highest-priority achievable Goal, and re-evaluates when a Nav gate changes or a schedule triggers. Game modules decide how a chosen Goal is carried out in loaded Cells.
_Avoid_: Task, objective, behavior, quest step

**Animation graph**:
Authored data, evaluated by the Engine, that blends and sequences a skeleton's animations according to parameters that Game modules set. Game modules also receive animation events from it.
_Avoid_: Animator, state machine, blend tree, rig logic

**Demo world**:
The first target World: a medieval village in a large forest in the mountains, used to showcase the Engine's features.
_Avoid_: Demo map, level, scene, sample
