#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Configure a cuSAT CPU-isolation profile in the active Jetson extlinux entry.

set -euo pipefail

readonly DEFAULT_CONFIG=/boot/extlinux/extlinux.conf
readonly PLATFORM_NAME=${CUSAT_JETSON_PLATFORM:-Thor}
readonly CPU_RANGE=${CUSAT_JETSON_CPU_RANGE:-4-11}
readonly PROFILE_COMMAND=${CUSAT_JETSON_PROFILE_COMMAND:-$0}
readonly DESIRED_ISOLCPUS="isolcpus=managed_irq,domain,$CPU_RANGE"
readonly DESIRED_RCU_NOCBS="rcu_nocbs=$CPU_RANGE"
readonly DESIRED_IRQAFFINITY="irqaffinity=${CUSAT_JETSON_IRQAFFINITY:-0-3}"

config=${CUSAT_EXTLINUX_CONF:-$DEFAULT_CONFIG}

usage() {
  cat <<'EOF'
Usage:
  boot-profile.sh check
  sudo boot-profile.sh apply
  sudo boot-profile.sh restore BACKUP

Commands:
  check    Inspect the DEFAULT extlinux entry and the running kernel.
  apply    Normalize only the DEFAULT entry and create a timestamped backup.
  restore  Restore an explicitly named backup after preserving the current file.

For fixture/testing use only, CUSAT_EXTLINUX_CONF may select another file.
The Orin wrapper also supplies its platform name and CPU range.
EOF
}

die() {
  echo "boot-profile: error: $*" >&2
  exit 1
}

require_root() {
  if [[ $EUID -ne 0 ]]; then
    die "this command changes the boot configuration; run it with sudo"
  fi
}

# Populates DEFAULT_LABEL, APPEND_LINE_NUMBER, and APPEND_TEXT for a config.
inspect_config() {
  local inspected=$1
  local -a defaults labels appends
  local label_start label_end

  [[ -f $inspected ]] || die "not a regular file: $inspected"
  [[ -r $inspected ]] || die "cannot read: $inspected"

  mapfile -t defaults < <(awk '$1 == "DEFAULT" { print $2 }' "$inspected")
  ((${#defaults[@]} == 1)) ||
    die "expected exactly one DEFAULT directive in $inspected; found ${#defaults[@]}"
  [[ -n ${defaults[0]} ]] || die "DEFAULT has no label in $inspected"
  DEFAULT_LABEL=${defaults[0]}

  mapfile -t labels < <(
    awk -v wanted="$DEFAULT_LABEL" '$1 == "LABEL" && $2 == wanted { print NR }' \
      "$inspected"
  )
  ((${#labels[@]} == 1)) ||
    die "expected exactly one LABEL $DEFAULT_LABEL in $inspected; found ${#labels[@]}"
  label_start=${labels[0]}

  label_end=$(awk -v start="$label_start" \
    'NR > start && $1 == "LABEL" { print NR; exit }' "$inspected")
  if [[ -z $label_end ]]; then
    label_end=$(( $(wc -l < "$inspected") + 1 ))
  fi

  mapfile -t appends < <(
    awk -v start="$label_start" -v stop="$label_end" \
      'NR > start && NR < stop && $1 == "APPEND" { print NR }' "$inspected"
  )
  ((${#appends[@]} == 1)) ||
    die "DEFAULT entry '$DEFAULT_LABEL' must contain exactly one APPEND line; found ${#appends[@]}"

  APPEND_LINE_NUMBER=${appends[0]}
  APPEND_TEXT=$(sed -n "${APPEND_LINE_NUMBER}p" "$inspected")
}

normalize_append() {
  local line=$1 indent body token
  local -a fields kept

  # Preserve indentation and every argument not owned by this profile.
  indent=${line%%[![:space:]]*}
  body=${line#"$indent"}
  read -r -a fields <<< "$body"
  [[ ${fields[0]:-} == APPEND ]] || die "selected line is not an APPEND directive"

  kept=(APPEND)
  for token in "${fields[@]:1}"; do
    case $token in
      isolcpus=*|nohz_full=*|rcu_nocbs=*|irqaffinity=*) ;;
      *) kept+=("$token") ;;
    esac
  done
  kept+=("$DESIRED_ISOLCPUS" "$DESIRED_RCU_NOCBS" "$DESIRED_IRQAFFINITY")

  printf '%s%s' "$indent" "${kept[*]}"
}

kernel_has_argument() {
  local command_line=$1 argument=$2
  [[ " $command_line " == *" $argument "* ]]
}

print_status() {
  local normalized running=''
  normalized=$(normalize_append "$APPEND_TEXT")

  echo "extlinux file : $config"
  echo "DEFAULT label : $DEFAULT_LABEL"
  echo "APPEND line   : $APPEND_LINE_NUMBER"
  if [[ $APPEND_TEXT == "$normalized" ]]; then
    echo "boot profile  : configured"
  else
    echo "boot profile  : needs normalization"
    echo "desired line  : $normalized"
  fi

  if [[ -r /proc/cmdline ]]; then
    running=$(< /proc/cmdline)
    if kernel_has_argument "$running" "$DESIRED_ISOLCPUS" &&
       kernel_has_argument "$running" "$DESIRED_RCU_NOCBS" &&
       kernel_has_argument "$running" "$DESIRED_IRQAFFINITY" &&
       [[ ! " $running " =~ [[:space:]]nohz_full= ]]; then
      echo "running kernel: configured"
    else
      echo "running kernel: not using the complete $PLATFORM_NAME cuSAT profile (apply, then reboot)"
    fi
  fi

  if [[ -r /sys/devices/system/cpu/isolated ]]; then
    echo "isolated CPUs : $(< /sys/devices/system/cpu/isolated)"
  fi
  if [[ -r /sys/devices/system/cpu/nohz_full ]]; then
    local nohz
    nohz=$(< /sys/devices/system/cpu/nohz_full)
    echo "nohz_full CPUs: ${nohz:-none (expected on $PLATFORM_NAME)}"
  fi
}

validate_normalized_file() {
  local candidate=$1 saved_config=$config normalized
  config=$candidate
  inspect_config "$config"
  normalized=$(normalize_append "$APPEND_TEXT")
  [[ $APPEND_TEXT == "$normalized" ]] || die "generated configuration failed validation"
  config=$saved_config
}

apply_profile() {
  local normalized backup temporary
  require_root
  inspect_config "$config"
  normalized=$(normalize_append "$APPEND_TEXT")

  if [[ $APPEND_TEXT == "$normalized" ]]; then
    echo "boot-profile: DEFAULT entry '$DEFAULT_LABEL' is already configured; no file changed"
    print_status
    return
  fi

  backup="${config}.cusat-backup.$(date +%Y%m%d-%H%M%S)"
  [[ ! -e $backup ]] || die "backup path unexpectedly exists: $backup"
  cp -a -- "$config" "$backup"

  temporary=$(mktemp "${config}.cusat.XXXXXX")
  trap 'rm -f -- "${temporary:-}"' EXIT
  awk -v target="$APPEND_LINE_NUMBER" -v replacement="$normalized" \
    'NR == target { $0 = replacement } { print }' "$config" > "$temporary"
  chmod --reference="$config" "$temporary"
  chown --reference="$config" "$temporary"
  if command -v chcon >/dev/null 2>&1; then
    chcon --reference="$config" "$temporary" 2>/dev/null || true
  fi

  validate_normalized_file "$temporary"
  mv -f -- "$temporary" "$config"
  trap - EXIT
  sync -f "$config" 2>/dev/null || sync

  echo "boot-profile: normalized DEFAULT entry '$DEFAULT_LABEL' only"
  echo "boot-profile: backup saved as $backup"
  echo "boot-profile: reboot $PLATFORM_NAME, then run '$PROFILE_COMMAND check'"
}

restore_backup() {
  local backup=$1 rescue temporary saved_config=$config
  require_root
  [[ -f $backup ]] || die "backup is not a regular file: $backup"
  [[ -r $backup ]] || die "cannot read backup: $backup"

  # Refuse arbitrary input: the backup itself must be a valid extlinux config.
  config=$backup
  inspect_config "$config"
  config=$saved_config

  rescue="${config}.before-restore.$(date +%Y%m%d-%H%M%S)"
  cp -a -- "$config" "$rescue"
  temporary=$(mktemp "${config}.restore.XXXXXX")
  trap 'rm -f -- "${temporary:-}"' EXIT
  cp -a -- "$backup" "$temporary"
  mv -f -- "$temporary" "$config"
  trap - EXIT
  sync -f "$config" 2>/dev/null || sync

  echo "boot-profile: restored $backup"
  echo "boot-profile: previous configuration preserved as $rescue"
  echo "boot-profile: reboot $PLATFORM_NAME to activate the restored entry"
}

action=${1:-check}
case $action in
  check)
    (($# == 1 || $# == 0)) || { usage >&2; exit 2; }
    inspect_config "$config"
    print_status
    ;;
  apply)
    (($# == 1)) || { usage >&2; exit 2; }
    apply_profile
    ;;
  restore)
    (($# == 2)) || { usage >&2; exit 2; }
    restore_backup "$2"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
