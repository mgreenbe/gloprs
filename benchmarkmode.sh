#!/bin/bash
# macOS benchmark preparation. Compatible with the system Bash 3.2.
# State is data, never sourced as shell code. Run as your login user, not sudo.
set -u
set -o pipefail
export LC_ALL=C
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
umask 077

usage() {
    cat <<'HELP'
Usage: ./benchmarkmode.sh --on|--off|--status [--dry-run]

--on       Disable currently enabled Spotlight stores; disable automatic Time
           Machine backups and stop an active backup; pause running user cloud,
           Photos, indexing and analysis workers; prevent idle system sleep.
--off      Resume only workers paused here, release the sleep assertion, and
           restore the recorded indexing/automatic-backup settings.
--status   Show the restore journal and current indexing state.
--dry-run  Preview --on/--off without writing state or changing services.

Run without sudo; administrative commands request sudo when needed.
State: ~/Library/Application Support/highsrs/benchmarkmode
Repeated --on preserves the original settings and catches newly started workers.
Run --off after benchmarking, including after an interrupted --on or a reboot.

Pausing cloud/FileProvider workers can make cloud-backed files and apps wait.
Keep benchmark inputs local, close Photos, and allow existing sync work to finish
before --on. An interrupted Time Machine backup is not forcibly restarted by
--off; its automatic schedule is restored. macOS can refuse protected processes;
failures are reported, and failed restores remain in the journal for retry.
This is best-effort quieting, not exclusive CPU isolation. New workers may start
later; run --on again if needed. Power profiles and networking stay unchanged.
HELP
}
fail() { printf 'benchmarkmode: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; failures=$((failures + 1)); }
note() { printf '%s\n' "$*"; }

mode= dry=0 failures=0 locked=0 pending_awake=
for arg in "$@"; do
    case "$arg" in
        --on|--off|--status) [ -z "$mode" ] || fail 'Choose one mode.'; mode=$arg ;;
        --dry-run) dry=1 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "Unknown argument: $arg" ;;
    esac
done
[ -n "$mode" ] || { usage >&2; exit 1; }
[ "$(/usr/bin/uname -s)" = Darwin ] || fail 'This script requires macOS.'
[ "$(/usr/bin/id -u)" != 0 ] || fail 'Run as your login user without sudo.'
uid=$(/usr/bin/id -u)
base="$HOME/Library/Application Support/highsrs"
state="$base/benchmarkmode"
lock="$base/benchmarkmode.lock"
# These are user workers, not launchd, the kernel, security or update services.
workers=(bird cloudd fileproviderd cloudphotod photolibraryd photoanalysisd
    mediaanalysisd corespotlightd mdworker mdworker_shared mdbulkimport
    suggestd knowledge-agent spotlightknowledged)

identity() {
    local pid=$1 started exe owner
    case "$pid" in ''|*[!0-9]*) return 1 ;; esac
    started=$(/bin/ps -p "$pid" -o lstart= 2>/dev/null) || return 1
    exe=$(/bin/ps -p "$pid" -o comm= 2>/dev/null) || return 1
    owner=$(/bin/ps -p "$pid" -o uid= 2>/dev/null) || return 1
    # Normalize ps padding; preserve spaces inside timestamps and executable paths.
    started=$(printf '%s' "$started" | sed 's/^ *//;s/ *$//')
    exe=$(printf '%s' "$exe" | sed 's/^ *//;s/ *$//')
    owner=$(printf '%s' "$owner" | tr -d ' ')
    [ -n "$started" ] && [ -n "$exe" ] && [ "$owner" = "$uid" ] || return 1
    case "$exe" in *$'\t'*|*$'\n'*) return 1 ;; esac
    printf '%s\t%s\t%s' "$started" "$exe" "$owner"
}
volume_uuid() {
    /usr/sbin/diskutil info -plist "$1" 2>/dev/null |
        /usr/bin/plutil -extract VolumeUUID raw -o - - 2>/dev/null
}
boot_id() { /usr/sbin/sysctl -n kern.bootsessionuuid; }
check_state() {
    [ ! -L "$state" ] && [ -d "$state" ] || fail 'Invalid restore directory.'
    [ "$(/usr/bin/stat -f %u "$state")" = "$uid" ] || fail 'Restore directory has another owner.'
    [ "$(cat "$state/version" 2>/dev/null)" = 1 ] || fail 'Unknown restore journal version.'
}
cleanup() {
    if [ -n "$pending_awake" ]; then /bin/kill -TERM "$pending_awake" 2>/dev/null || :; fi
    if [ "$locked" = 1 ]; then rm -f "$lock/owner"; rmdir "$lock" 2>/dev/null || :; fi
}
acquire_lock() {
    mkdir -p "$base" || fail 'Cannot create state parent.'
    [ ! -L "$base" ] && [ "$(/usr/bin/stat -f %u "$base")" = "$uid" ] || fail 'Invalid state parent.'
    if ! mkdir "$lock" 2>/dev/null; then
        [ ! -L "$lock" ] && [ -d "$lock" ] || fail 'Invalid lock.'
        local owner saved current original latest retired
        IFS=$'\t' read -r owner saved < "$lock/owner" || fail "Lock has no owner; inspect $lock before removing a stale lock."
        case "$owner" in ''|*[!0-9]*) fail 'Invalid lock owner.' ;; esac
        current=$(identity "$owner") || current=gone
        [ "$current" != "$saved" ] || fail 'Another invocation is active.'
        # Serialize stale-lock recovery too. Re-read the owner after claiming
        # recovery: another invocation may already have replaced the old lock.
        original=$(printf '%s\t%s' "$owner" "$saved")
        mkdir "$lock/reclaim" 2>/dev/null || fail 'Another invocation is recovering the lock.'
        latest=$(cat "$lock/owner" 2>/dev/null) || latest=
        if [ "$latest" != "$original" ]; then
            rmdir "$lock/reclaim" 2>/dev/null || :
            fail 'Lock owner changed; retry.'
        fi
        retired="$lock.retired.$$"
        [ ! -e "$retired" ] || fail "Inspect stale recovery directory: $retired"
        # Rename the entire claimed directory atomically; never delete a newly
        # created lock at the original path while cleaning up the stale one.
        mv "$lock" "$retired" || fail 'Cannot retire stale lock.'
        rm -f "$retired/owner" && rmdir "$retired/reclaim" && rmdir "$retired" || fail 'Cannot clean retired lock.'
        mkdir "$lock" 2>/dev/null || fail 'Another invocation acquired the lock; retry.'
    fi
    locked=1
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    printf '%s\t%s\n' "$$" "$(identity "$$")" > "$lock/owner" || fail 'Cannot record lock.'
}

if [ "$mode" = --status ]; then
    if [ -e "$state" ] || [ -L "$state" ]; then
        check_state
        note "Restore journal: $state"
        for file in spotlight processes awake time-machine; do
            if [ -s "$state/$file" ]; then note "[$file]"; cat "$state/$file"; fi
        done
        note 'A journal records requested changes; check warnings from --on/--off.'
    else
        note 'Benchmark mode has no restore journal.'
    fi
    /usr/bin/mdutil -a -s
    exit $?
fi

if [ "$dry" = 0 ]; then acquire_lock; fi
if [ "$mode" = --on ]; then
    if [ -e "$state" ] || [ -L "$state" ]; then
        check_state
        [ "$(cat "$state/boot")" = "$(boot_id)" ] || fail 'Journal is from a previous boot; run --off first.'
    fi
    # Authenticate before pausing anything. --off resumes user workers first,
    # even if administrative restoration subsequently fails or is interrupted.
    if [ "$dry" = 0 ]; then
        /usr/bin/sudo -v || fail 'Administrative access is needed for indexing/backups.'
        if [ ! -d "$state" ]; then
            boot=$(boot_id) || fail 'Cannot identify this boot.'
            mkdir "$state" || fail 'Cannot create restore journal.'
            printf '1\n' > "$state/version"
            printf '%s\n' "$boot" > "$state/boot"
        fi
    fi
    stores=$(/usr/bin/mdutil -a -s 2>&1) || { warn "Cannot inspect Spotlight: $stores"; stores=; }
    volume=
    while IFS= read -r line; do
        case "$line" in
            /*:) volume=${line%:} ;;
            *'Indexing enabled.'*)
                [ -n "$volume" ] || continue
                case "$volume" in *$'\t'*|*$'\n'*) warn 'Skipping unrepresentable volume name.'; continue ;; esac
                uuid=$(volume_uuid "$volume") || { warn "No volume UUID: $volume"; continue; }
                if [ "$dry" = 1 ]; then note "Would disable Spotlight: $volume ($uuid)"; continue; fi
                # Write ahead: an interruption after mdutil must still be recoverable.
                record=$(printf '%s\t%s' "$volume" "$uuid")
                if ! grep -Fqx -- "$record" "$state/spotlight" 2>/dev/null; then
                    printf '%s\n' "$record" >> "$state/spotlight" || fail 'Cannot journal Spotlight.'
                fi
                /usr/bin/sudo /usr/bin/mdutil -i off "$volume" || warn "Could not disable Spotlight: $volume"
                ;;
        esac
    done <<< "$stores"
    auto=$(/usr/bin/defaults read /Library/Preferences/com.apple.TimeMachine AutoBackup 2>/dev/null) || auto=unknown
    if [ "$auto" = 1 ]; then
        if [ "$dry" = 1 ]; then note 'Would disable automatic Time Machine backups.'
        else
            printf 'enabled\n' > "$state/time-machine" || fail 'Cannot journal Time Machine.'
            /usr/bin/sudo /usr/bin/tmutil disable || warn 'Could not disable automatic backups.'
        fi
    elif [ "$auto" = unknown ]; then
        note 'Time Machine AutoBackup is unset/unreadable; leaving that setting unchanged.'
    fi
    running=$(/usr/bin/tmutil status 2>/dev/null | awk '$1 == "Running" {gsub(/;/,"",$3); print $3}') || running=unknown
    if [ "$running" = 1 ]; then
        if [ "$dry" = 1 ]; then note 'Would stop the active Time Machine backup.'
        else /usr/bin/sudo /usr/bin/tmutil stopbackup || warn 'Could not stop the active backup.'; fi
    fi
    for name in "${workers[@]}"; do
        pids=$(/usr/bin/pgrep -u "$uid" -x "$name" 2>/dev/null) || pids=
        for pid in $pids; do
            ident=$(identity "$pid") || continue
            exe=$(printf '%s' "$ident" | cut -f2)
            [ "${exe##*/}" = "$name" ] || continue
            status=$(/bin/ps -p "$pid" -o state= 2>/dev/null) || continue
            case "$status" in *T*) continue ;; esac # Never take ownership of an existing stop.
            if [ "$dry" = 1 ]; then note "Would pause $name (PID $pid)."; continue; fi
            record=$(printf '%s\t%s' "$pid" "$ident")
            if ! grep -Fqx -- "$record" "$state/processes" 2>/dev/null; then
                printf '%s\n' "$record" >> "$state/processes" || fail 'Cannot journal process.'
            fi
            if /bin/kill -STOP "$pid"; then note "Paused $name (PID $pid)."
            else
                warn "macOS refused to pause $name (PID $pid)."
                # A failed STOP needs no CONT; do not retain an unrestorable entry.
                grep -Fvx -- "$record" "$state/processes" > "$state/processes.pending" || :
                mv "$state/processes.pending" "$state/processes" || fail 'Cannot update process journal.'
            fi
        done
    done
    if [ "$dry" = 0 ] && [ -s "$state/awake" ]; then
        IFS=$'\t' read -r awake saved < "$state/awake"
        ident=$(identity "$awake") || ident=gone
        [ "$ident" = "$saved" ] || rm "$state/awake"
    fi
    if [ "$dry" = 1 ]; then note 'Would start/retain an idle-sleep assertion (caffeinate -i).'
    elif [ ! -s "$state/awake" ]; then
        # nohup lets the assertion survive this script and its terminal exiting.
        /usr/bin/nohup /usr/bin/caffeinate -i </dev/null >/dev/null 2>&1 &
        awake=$!
        pending_awake=$awake
        ready=0
        for attempt in 1 2 3 4 5 6 7 8 9 10; do
            ident=$(identity "$awake") || ident=
            exe=$(printf '%s' "$ident" | cut -f2)
            if [ "$exe" = /usr/bin/caffeinate ]; then ready=1; break; fi
            sleep 0.1
        done
        if [ "$ready" = 1 ]; then
            printf '%s\t%s\n' "$awake" "$ident" > "$state/awake" || fail 'Cannot journal caffeinate.'
            pending_awake=
        else warn 'Could not start the sleep assertion.'; fi
    fi
else
    if [ ! -e "$state" ] && [ ! -L "$state" ]; then note 'Nothing to restore.'; exit 0; fi
    check_state
    current_boot=$(boot_id) || fail 'Cannot identify current boot; keep the journal and retry --off.'
    [ -n "$current_boot" ] || fail 'Empty boot identity; keep the journal and retry --off.'
    same_boot=0
    [ "$(cat "$state/boot")" = "$current_boot" ] && same_boot=1
    for file in processes awake; do
        [ -s "$state/$file" ] || continue
        if [ "$dry" = 0 ]; then : > "$state/$file.pending" || fail 'Cannot prepare process restore journal.'; fi
        while IFS=$'\t' read -r pid saved; do
            case "$pid" in
                ''|*[!0-9]*)
                    warn "Invalid PID record in $file; keeping it for inspection."
                    if [ "$dry" = 0 ]; then
                        printf '%s\t%s\n' "$pid" "$saved" >> "$state/$file.pending" || fail 'Cannot retain invalid record; original journal kept.'
                    fi
                    continue ;;
            esac
            ident=$(identity "$pid") || ident=gone
            if [ "$same_boot" = 0 ] || [ "$ident" != "$saved" ]; then continue; fi
            signal=CONT; [ "$file" = awake ] && signal=TERM
            if [ "$dry" = 1 ]; then note "Would send $signal to recorded PID $pid."
            elif ! /bin/kill -"$signal" "$pid"; then
                warn "Could not restore PID $pid."
                printf '%s\t%s\n' "$pid" "$saved" >> "$state/$file.pending" || fail 'Cannot retain failed process restore; original journal kept.'
            fi
        done < "$state/$file"
        if [ "$dry" = 0 ]; then mv "$state/$file.pending" "$state/$file" || fail 'Cannot commit process restore journal.'; fi
    done
    if [ -s "$state/spotlight" ]; then
        if [ "$dry" = 0 ]; then : > "$state/spotlight.pending" || fail 'Cannot prepare Spotlight restore journal.'; fi
        while IFS=$'\t' read -r volume saved; do
            uuid=$(volume_uuid "$volume") || uuid=missing
            if [ "$uuid" != "$saved" ]; then
                warn "Volume missing or replaced; remount it and retry --off: $volume"
                if [ "$dry" = 0 ]; then
                    printf '%s\t%s\n' "$volume" "$saved" >> "$state/spotlight.pending" || fail 'Cannot retain Spotlight restore; original journal kept.'
                fi
            elif [ "$dry" = 1 ]; then note "Would re-enable Spotlight: $volume"
            elif ! /usr/bin/sudo /usr/bin/mdutil -i on "$volume"; then
                warn "Could not restore Spotlight: $volume"
                printf '%s\t%s\n' "$volume" "$saved" >> "$state/spotlight.pending" || fail 'Cannot retain Spotlight restore; original journal kept.'
            fi
        done < "$state/spotlight"
        if [ "$dry" = 0 ]; then mv "$state/spotlight.pending" "$state/spotlight" || fail 'Cannot commit Spotlight restore journal.'; fi
    fi
    if [ -s "$state/time-machine" ]; then
        if [ "$dry" = 1 ]; then note 'Would restore automatic Time Machine backups.'
        elif /usr/bin/sudo /usr/bin/tmutil enable; then rm "$state/time-machine" || warn 'Backup schedule restored, but journal cleanup failed.'
        else warn 'Could not restore automatic backups.'; fi
    fi
    if [ "$dry" = 0 ]; then
        remaining=0
        for file in spotlight processes awake time-machine; do
            [ ! -s "$state/$file" ] || remaining=1
        done
        if [ "$remaining" = 0 ]; then
            rm -f "$state/version" "$state/boot" "$state/spotlight" "$state/processes" "$state/awake"
            rmdir "$state" || warn 'Restore directory contains unexpected files; inspect it.'
        fi
    fi
fi
if [ "$failures" -gt 0 ]; then
    note "Completed with $failures warning(s); review them and retry restoration if needed."
    exit 2
fi
[ "$dry" = 0 ] || { note 'Dry run: nothing changed.'; exit 0; }
if [ "$mode" = --on ]; then
    note 'Benchmark preparation complete. Run --off after benchmarking.'
else
    note 'Restoration complete.'
fi
