
## Profiling utilities

The library contains several model profiling infrastructure components within `::vpux::profiling` namespace.

1. Profiling output parser `parser/api.hpp`
2. Reporting code and profiling hooks used also by [Compiler Schedule Trace](../../../../guides/how-to-get-schedule-trace-and-analysis.md) `reports/api.hpp`
3. Metadata serialization/deserialization code shared between the compiler and parser `metadata.hpp`
4. Profiling utilities
    - definitions shared with the compiler `location.hpp` `common.hpp`
    - TaskInfo interface used between parser and reporting `taskinfo.hpp`
    - task name parsers

The components are small and therefore are grouped into a single static library while preserving separation of subcomponent interfaces via separate header files.
Both profiling output and metadata formats lead to version dependency between the compiler and the parser, therefore profiling parser must be distributed (or statically linked) in the same binary package.

Users of this library include:
1. vpux_compiler
2. vpux_plugin (used primarily for the compiler in plugin configuration)
3. vpux_driver_compiler
4. Standalone profiling parser `prof_parser`
