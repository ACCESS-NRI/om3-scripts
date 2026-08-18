# modify_restart_date.py

Rebrand a payu ACCESS-OM3 configuration's restart set to a new date, so it
can be used to start a new run/experiment on that date.

Run from inside a payu configuration directory (i.e. the directory
containing `config.yaml`). The script:

1. Reads the restart set to rebrand from config.yaml's existing `restart:`
   field.
2. Writes the rebranded restart set to an `initial_restart` subdirectory of
   the configuration directory (added to `.gitignore` - restart netCDFs are
   not committed).
3. Updates config.yaml's `restart:` field to point at `initial_restart`.
4. Commits the config.yaml (and `.gitignore`) change to the configuration
   directory's git repo, with provenance details (script version, command
   used, and the original restart files' md5 hashes) in the commit message.

The original restart set (config.yaml's old `restart:` value) is never
modified.

The current restart date is auto-detected from the filenames in that
restart set (it errors out if more than one date is found there, or if a
ww3 restart is present - see Limitations below).

## Requirements

A Python environment with `netCDF4` installed, e.g. on Gadi:

```
module use /g/data/xp65/public/modules
module load conda/analysis3
```

## Usage

```
cd /path/to/your/access-om3-configs
python3 modify_restart_date.py --new_date <YYYY-MM-DD>
```

- `--new_date`: the date the rebranded restart should represent, e.g.
  `1958-01-01` (default: `1958-01-01`).
- `--config_dir`: path to the payu configuration directory (default: the
  current directory).

### Example

Config.yaml already has:

```
restart: /g/data/ol01/outputs/access-om3-25km/MC_25km_jra_iaf+wombatlite-test4-d28e0359/restart065
```

(a restart dated 2024-01-01). To rebrand it to start a new run at
1958-01-01 instead:

```
cd /path/to/your/access-om3-configs
python3 modify_restart_date.py --new_date 1958-01-01
```

This writes the rebranded restart to `./initial_restart`, updates
`config.yaml`'s `restart:` field to point at it, and commits that change.
Then run `payu sweep && payu run` as usual.

## What gets changed

- `cpl` and `cice` restarts: renamed, and their internal date metadata
  (`start_ymd`/`curr_ymd`/`start_tod`/`curr_tod` for `cpl`;
  `myear`/`mmonth`/`mday`/`msec` for `cice`) is rewritten to the new date.
- All other restart files (`mom6`, `datm`, `drof`, including uncollated
  per-tile `*.nc.NNNN` fragments, and undated files such as FMS-style
  `*.res.nc.NNNN` tracer restarts) are copied through, renamed only where
  the old date appears in their filename.
- `rpointer.*` files are rewritten to point at the renamed files.

## Limitations

- **WW3 is not supported.** If a ww3 restart file or `rpointer.wav` is
  found in the restart set, the script raises an error rather than
  silently mishandling it. WW3 restarts and their `ww3_in` `initfile`
  setting need to be handled separately.
- **datm/drof stream bookkeeping is not updated.** These restarts contain a
  per-stream `date`/`timeofday` array recording which forcing input files
  were last read, which still reflects the *original* forcing period. This
  is fine if the new run reuses the same forcing streams/config, but check
  the new case's `datm_in`/`drof_in` stream setup if moving to a very
  different calendar period with different forcing data.
- **CICE's `istep1` step counter is not reset.** It's carried over from
  the original restart rather than reset to 0, so anything keyed off raw
  step counts (rather than calendar date) inherits the original run's step
  count. Calendar-based history/restart frequencies (e.g. `histfreq =
  "d","m"`, `dumpfreq = "x"`) are unaffected.
- You still need to update the new case's own start date configuration
  (e.g. `nuopc.runconfig`) to match `--new_date`.
- `--config_dir` must already be a git repository with `config.yaml` in
  its top level, and `initial_restart` must not already exist there.
