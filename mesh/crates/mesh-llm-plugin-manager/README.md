# mesh-llm-plugin-manager

Plugin package management primitives for Mesh LLM.

This crate owns install-reference parsing, platform target naming, native
release asset selection, validation of packaged plugin manifest metadata
(including local web UI bundle roots and entry scripts), and local
installed-plugin metadata. Catalog/GitHub installs use `install_plugin`;
authoring and release validation can pass a `.tar.gz` or `.zip` directly to
`install_plugin_archive`, which uses the same extraction and package-validation
boundary. A downloaded release asset must be exactly the size GitHub reports,
at most 512 MiB, and extraction refuses an archive that unpacks to more than
1 GiB or 10,000 entries. It deliberately
does not render CLI output; callers should render progress and status events in
the host application.

