#![expect(
    clippy::cast_possible_truncation,
    reason = "Synthetic fixtures use bounded sample counts and intentionally quantized audio; exact comparisons verify unchanged settings and samples."
)]

use super::*;

fn plan() -> CaptureSessionPlan {
    serde_json::from_str(include_str!("../../tests/fixtures/capture-session.json")).unwrap()
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "Microphone indices below 33 convert to f64 survey coordinates exactly."
)]
fn supported_microphone_counts_round_trip() {
    for count in [2_usize, 4, 8, 32] {
        let mut value = plan();
        for index in 2..count {
            let mut mic = value.microphones[0].clone();
            mic.id = format!("mic-{index}");
            mic.device = format!("device-{index}");
            if let Some(position) = mic.position_m.as_mut() {
                position[0] = index as f64;
            }
            value.microphones.push(mic);
        }
        let session = value.validate().unwrap();
        let json = serde_json::to_string(session.plan()).unwrap();
        let restored: CaptureSessionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.validate().unwrap().plan().microphones.len(), count);
    }
    let mut value = plan();
    for index in 2..33_usize {
        let mut mic = value.microphones[0].clone();
        mic.id = format!("mic-{index}");
        mic.device = format!("device-{index}");
        if let Some(position) = mic.position_m.as_mut() {
            position[0] = index as f64;
        }
        value.microphones.push(mic);
    }
    assert!(value.validate().is_err());
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "This end-to-end fixture keeps setup, execution, and artifact assertions together."
)]
fn invalid_plans_fail_before_hardware_access() {
    let cases: &[fn(&mut CaptureSessionPlan)] = &[
        |p| p.version = 2,
        |p| p.sample_rate_hz = 16_000,
        |p| p.output_device.clear(),
        |p| p.microphones.truncate(1),
        |p| p.microphones[1].id = p.microphones[0].id.clone(),
        |p| p.microphones[1].device = p.microphones[0].device.clone(),
        |p| p.microphones[0].calibration_file.clear(),
        |p| p.microphones[0].gain_db = f64::NAN,
        |p| p.microphones[0].input_channel = 64,
        |p| {
            if let Some(position) = p.microphones[0].position_m.as_mut() {
                position[0] = f64::INFINITY;
            }
        },
        |p| p.microphones[0].position_uncertainty_mm = Some(-1.0),
        |p| p.sources.clear(),
        |p| p.sources[0].output_channel = 64,
        |p| p.sources.push(p.sources[0].clone()),
        |p| p.repeat_count = 0,
        |p| p.repeat_count = 17,
        |p| p.sweep.duration_secs = f64::INFINITY,
        |p| p.sweep.duration_secs = 1e100,
        |p| p.sweep.duration_secs = 1e-100,
        |p| p.sweep.start_hz = p.sweep.end_hz,
        |p| p.sweep.end_hz = 24_000.0,
        |p| p.sweep.amplitude = 1.01,
        |p| p.sweep.amplitude = 0.0,
        |p| p.sweep.amplitude = 1e-100,
        |p| {
            p.room_dimensions = Some(RoomDimensions {
                length: 0.0,
                width: 4.0,
                height: 2.5,
            });
        },
        |p| {
            p.room_dimensions = Some(RoomDimensions {
                length: 5.0,
                width: -4.0,
                height: 2.5,
            });
        },
        |p| {
            p.room_dimensions = Some(RoomDimensions {
                length: 5.0,
                width: 4.0,
                height: f64::NAN,
            });
        },
        |p| {
            p.room_dimensions = Some(RoomDimensions {
                length: f64::INFINITY,
                width: 4.0,
                height: 2.5,
            });
        },
        |p| {
            p.room_dimensions = Some(RoomDimensions {
                length: 1e200,
                width: 1e200,
                height: 1e200,
            });
        },
        |p| p.setup_description = Some(String::new()),
        |p| p.setup_description = Some("   ".into()),
        |p| {
            p.probe = Some(CaptureProbe {
                probe_duration_ms: 0.0,
                silence_duration_ms: 500.0,
                signal_level_db: -6.0,
            });
        },
        |p| {
            p.probe = Some(CaptureProbe {
                probe_duration_ms: 1000.0,
                silence_duration_ms: -1.0,
                signal_level_db: -6.0,
            });
        },
        |p| {
            p.probe = Some(CaptureProbe {
                probe_duration_ms: 1000.0,
                silence_duration_ms: 500.0,
                signal_level_db: 1.0,
            });
        },
        |p| {
            p.bass_anchor = Some(CaptureBassAnchor {
                bass_freq_hz: 24000.0,
                bass_duration_s: 2.0,
                fade_ms: 50.0,
                num_windows: 8,
                silence_duration_ms: 500.0,
                signal_level_db: -6.0,
            });
        },
        |p| {
            p.bass_anchor = Some(CaptureBassAnchor {
                bass_freq_hz: 30.0,
                bass_duration_s: 0.05,
                fade_ms: 50.0,
                num_windows: 8,
                silence_duration_ms: 500.0,
                signal_level_db: -6.0,
            });
        },
        |p| {
            p.bass_anchor = Some(CaptureBassAnchor {
                bass_freq_hz: 30.0,
                bass_duration_s: 2.0,
                fade_ms: 50.0,
                num_windows: 0,
                silence_duration_ms: 500.0,
                signal_level_db: -6.0,
            });
        },
        |p| {
            p.loopback = Some(CaptureLoopback {
                device: p.microphones[0].device.clone(),
                input_channel: p.microphones[0].input_channel,
            });
        },
        |p| {
            p.bass_anchor = Some(CaptureBassAnchor {
                bass_freq_hz: 30.0,
                bass_duration_s: 2.0,
                fade_ms: 50.0,
                num_windows: 8,
                silence_duration_ms: 500.0,
                signal_level_db: -6.0,
            });
            p.loopback = Some(CaptureLoopback {
                device: "other-device".into(),
                input_channel: 3,
            });
        },
        |p| p.sweep.bass_octave_duration_s = Some(0.5),
        |p| p.sweep.bass_octave_duration_s = Some(11.0),
        |p| p.sweep.bass_octave_duration_s = Some(f32::NAN),
        |p| {
            p.loopback = Some(CaptureLoopback {
                device: p.microphones[0].device.clone(),
                input_channel: 7,
            });
            p.ctc = Some(CaptureCtc {
                left_ear: "seat-left".into(),
                right_ear: "seat-ghost".into(),
                head_position: CaptureHeadPosition {
                    id: "head-0".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                },
            });
        },
        |p| {
            p.loopback = Some(CaptureLoopback {
                device: p.microphones[0].device.clone(),
                input_channel: 7,
            });
            p.ctc = Some(CaptureCtc {
                left_ear: "seat-left".into(),
                right_ear: "seat-left".into(),
                head_position: CaptureHeadPosition {
                    id: "head-0".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                },
            });
        },
        |p| {
            p.ctc = Some(CaptureCtc {
                left_ear: "seat-left".into(),
                right_ear: "seat-right".into(),
                head_position: CaptureHeadPosition {
                    id: "head-0".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                },
            });
        },
        |p| {
            p.loopback = Some(CaptureLoopback {
                device: p.microphones[0].device.clone(),
                input_channel: 7,
            });
            p.ctc = Some(CaptureCtc {
                left_ear: "seat-left".into(),
                right_ear: "seat-right".into(),
                head_position: CaptureHeadPosition {
                    id: "  ".into(),
                    x: 0.0,
                    y: 0.0,
                    z: f64::NAN,
                    yaw_deg: 0.0,
                },
            });
        },
        |p| p.sources[0].position_m = Some([0.0, 0.0, f64::NAN]),
        |p| {
            p.sources[0].position_m = Some([1.0, 2.0, 3.0]);
            p.sources[0].position_uncertainty_mm = None;
        },
        |p| {
            p.sources[0].position_m = None;
            p.sources[0].position_uncertainty_mm = Some(5.0);
        },
        |p| {
            p.sources[0].position_m = Some([1.0, 2.0, 3.0]);
            p.sources[0].position_uncertainty_mm = Some(-1.0);
        },
        |p| p.sources[0].source_delay_calibration_ms = Some(f64::NAN),
        |p| p.sources[0].source_delay_calibration_ms = Some(-0.5),
        |p| p.sources[0].source_delay_calibration_ms = Some(100.5),
        |p| {
            p.sound_speed = Some(CaptureSoundSpeed {
                m_s: 200.0,
                uncertainty_m_s: 1.0,
                air_temperature_c: None,
            });
        },
        |p| {
            p.sound_speed = Some(CaptureSoundSpeed {
                m_s: 343.0,
                uncertainty_m_s: 343.0,
                air_temperature_c: None,
            });
        },
        |p| {
            p.sound_speed = Some(CaptureSoundSpeed {
                m_s: 343.0,
                uncertainty_m_s: 1.0,
                air_temperature_c: Some(f64::NAN),
            });
        },
        |p| p.localization_calibration_file = Some(PathBuf::new()),
    ];
    for (index, change) in cases.iter().enumerate() {
        let mut value = plan();
        change(&mut value);
        assert!(value.validate().is_err(), "invalid case {index} accepted");
    }
}

#[test]
fn room_dimensions_and_setup_description_round_trip() {
    let mut value = plan();
    value.room_dimensions = Some(RoomDimensions {
        length: 5.0,
        width: 4.0,
        height: 2.5,
    });
    value.setup_description = Some("Treated room; mic at main seat".into());
    let session = value.validate().unwrap();
    let json = serde_json::to_string(session.plan()).unwrap();
    assert!(json.contains("room_dimensions"));
    assert!(json.contains("setup_description"));
    let restored: CaptureSessionPlan = serde_json::from_str(&json).unwrap();
    let frozen = restored.validate().unwrap();
    assert_eq!(
        frozen.plan().room_dimensions,
        Some(RoomDimensions {
            length: 5.0,
            width: 4.0,
            height: 2.5,
        })
    );
    assert_eq!(
        frozen.plan().setup_description.as_deref(),
        Some("Treated room; mic at main seat")
    );
}

#[test]
fn timing_emitter_prefers_declared_reference_over_first_source() {
    let fallback = plan().validate().unwrap();
    assert_eq!(timing_emitter_channel(&fallback), 0);
    let summary = timing_emitter_summary(&fallback);
    assert!(summary.contains("output channel 0"), "{summary}");
    assert!(summary.contains("source 'left'"), "{summary}");
    assert!(summary.contains("arrival-aligned"), "{summary}");

    let mut declared = plan();
    declared.timing_reference = Some(CaptureTimingReference {
        output_channel: 5,
        position_m: [1.0, 2.0, 3.0],
        position_uncertainty_mm: 0.5,
        sound_speed_m_s: 343.0,
        sound_speed_uncertainty_m_s: 1.0,
    });
    let fixed = declared.validate().unwrap();
    assert_eq!(timing_emitter_channel(&fixed), 5);
    let summary = timing_emitter_summary(&fixed);
    assert!(summary.contains("output channel 5"), "{summary}");
    assert!(summary.contains("dedicated emitter"), "{summary}");
    assert!(summary.contains("[1.000, 2.000, 3.000]"), "{summary}");
}

#[test]
fn auxiliary_sections_round_trip_when_valid() {
    let mut value = plan();
    value.probe = Some(CaptureProbe {
        probe_duration_ms: 1000.0,
        silence_duration_ms: 500.0,
        signal_level_db: -6.0,
    });
    value.bass_anchor = Some(CaptureBassAnchor {
        bass_freq_hz: 30.0,
        bass_duration_s: 2.0,
        fade_ms: 50.0,
        num_windows: 8,
        silence_duration_ms: 500.0,
        signal_level_db: -6.0,
    });
    value.loopback = Some(CaptureLoopback {
        device: value.microphones[0].device.clone(),
        input_channel: 7,
    });
    let frozen = value.validate().unwrap();
    let restored: CaptureSessionPlan =
        serde_json::from_slice(&serde_json::to_vec(frozen.plan()).unwrap()).unwrap();
    let frozen = restored.validate().unwrap();
    assert!(frozen.plan().probe.is_some());
    assert!(frozen.plan().bass_anchor.is_some());
    assert_eq!(frozen.plan().loopback.as_ref().unwrap().input_channel, 7);
}

#[test]
fn storage_budget_prices_bass_weighted_sweeps() {
    let mut heavy = plan();
    heavy.sample_rate_hz = 192_000;
    heavy.sweep.duration_secs = 30.0;
    heavy.sweep.start_hz = 1.0;
    heavy.sweep.bass_octave_duration_s = Some(10.0);
    let error = heavy.validate().unwrap_err().to_string();
    assert!(error.contains("bounded capture storage"), "{error}");

    let mut plain = plan();
    plain.sample_rate_hz = 192_000;
    plain.sweep.duration_secs = 30.0;
    plain.validate().unwrap();

    let mut weighted = plan();
    weighted.sweep.bass_octave_duration_s = Some(3.0);
    let frozen = weighted.clone().validate().unwrap();
    assert_eq!(frozen.plan().sweep.bass_octave_duration_s, Some(3.0));
    let json = serde_json::to_string(&weighted).unwrap();
    assert!(json.contains("bass_octave_duration_s"));
}

#[test]
fn ctc_declaration_round_trips_when_valid() {
    let mut value = plan();
    value.microphones[1].device = value.microphones[0].device.clone();
    value.microphones[1].input_channel = 1;
    value.loopback = Some(CaptureLoopback {
        device: value.microphones[0].device.clone(),
        input_channel: 7,
    });
    value.ctc = Some(CaptureCtc {
        left_ear: "seat-left".into(),
        right_ear: "seat-right".into(),
        head_position: CaptureHeadPosition {
            id: "head-0".into(),
            x: 1.0,
            y: 2.0,
            z: 1.2,
            yaw_deg: 15.0,
        },
    });
    let frozen = value.clone().validate().unwrap();
    assert!(frozen.plan().ctc.is_some());
    let json = serde_json::to_string(&value).unwrap();
    assert!(json.contains("\"ctc\""));
    let restored: CaptureSessionPlan = serde_json::from_str(&json).unwrap();
    restored.validate().unwrap();
    value.microphones[1].device = "independent-clock".into();
    assert!(value.validate().is_err());
}

#[test]
fn shipped_ctc_examples_declare_a_common_input_clock() {
    for json in [
        include_str!("../../examples/ctc-head-position.json"),
        include_str!("../../examples/full-roomeq-session.json"),
    ] {
        let plan: CaptureSessionPlan = serde_json::from_str(json).unwrap();
        plan.validate().unwrap();
    }
}

#[test]
fn auxiliary_and_aggregate_memory_budgets_fail_during_plan_validation() {
    for (duration, gap) in [(1e20, 100.0), (100.0, 1e20)] {
        let mut value = plan();
        value.probe = Some(CaptureProbe {
            probe_duration_ms: duration,
            silence_duration_ms: gap,
            signal_level_db: -20.0,
        });
        assert!(value.validate().is_err());
    }
    let mut value = plan();
    value.bass_anchor = Some(CaptureBassAnchor {
        bass_freq_hz: 30.0,
        bass_duration_s: 1e20,
        fade_ms: 50.0,
        num_windows: 8,
        silence_duration_ms: 100.0,
        signal_level_db: -20.0,
    });
    assert!(value.validate().is_err());
    let mut value = plan();
    let template = value.microphones[0].clone();
    value.microphones = (0_u16..32)
        .map(|index| {
            let mut microphone = template.clone();
            microphone.id = format!("mic-{index}");
            microphone.input_channel = index;
            microphone.position_m = None;
            microphone.position_uncertainty_mm = None;
            microphone
        })
        .collect();
    value.geometry = CaptureGeometry::Spread;
    value.sample_rate_hz = 384_000;
    value.sweep.duration_secs = 10.0;
    let error = value.validate().unwrap_err().to_string();
    assert!(error.contains("512 MiB"), "{error}");
}

#[test]
fn legacy_plan_without_room_fields_stays_valid() {
    let json = serde_json::to_string(&plan()).unwrap();
    assert!(!json.contains("room_dimensions"));
    assert!(!json.contains("setup_description"));
    let frozen = plan().validate().unwrap();
    assert_eq!(frozen.plan().room_dimensions, None);
    assert_eq!(frozen.plan().setup_description, None);
}

#[test]
fn legacy_plan_defaults_to_one_repeat_and_repeat_count_round_trips() {
    let mut json = serde_json::to_value(plan()).unwrap();
    json.as_object_mut().unwrap().remove("repeat_count");
    let legacy: CaptureSessionPlan = serde_json::from_value(json).unwrap();
    assert_eq!(legacy.repeat_count, 1);

    let mut repeated = legacy;
    repeated.repeat_count = 3;
    let restored: CaptureSessionPlan =
        serde_json::from_slice(&serde_json::to_vec(&repeated).unwrap()).unwrap();
    assert_eq!(restored.validate().unwrap().plan().repeat_count, 3);
}

#[test]
fn repeat_count_keeps_the_total_take_inventory_bounded() {
    let mut value = plan();
    while value.sources.len() < 64 {
        let index = value.sources.len();
        value.sources.push(CaptureSource {
            id: format!("source-{index}"),
            output_channel: index as u16,
            position_m: None,
            position_uncertainty_mm: None,
            source_delay_calibration_ms: None,
        });
    }
    value.repeat_count = 9;
    assert!(
        value
            .validate()
            .unwrap_err()
            .to_string()
            .contains("1024-take")
    );
}

#[test]
fn compact_geometry_requires_precise_distinct_positions() {
    let mut value = plan();
    value.geometry = CaptureGeometry::Compact;
    value.microphones[1].position_m = Some([0.05, 0.0, 0.0]);
    assert!(value.clone().validate().is_ok());
    value.microphones[1].position_uncertainty_mm = Some(1.01);
    assert!(value.clone().validate().is_err());
    value.microphones[1].position_uncertainty_mm = Some(1.0);
    value.microphones[1].position_m = value.microphones[0].position_m;
    assert!(value.validate().is_err());
}

#[test]
fn unsurveyed_microphones_validate_as_a_pair() {
    // Both absent: an inverse/joint session plan is valid.
    let mut value = plan();
    for mic in &mut value.microphones {
        mic.position_m = None;
        mic.position_uncertainty_mm = None;
    }
    assert!(value.clone().validate().is_ok());
    // Split pairs refuse, mirroring the source rule.
    value.microphones[0].position_m = Some([0.0, 0.0, 1.0]);
    assert!(value.clone().validate().is_err());
    value.microphones[0].position_m = None;
    value.microphones[0].position_uncertainty_mm = Some(1.0);
    assert!(value.clone().validate().is_err());
    // Two unsurveyed compact mics are not duplicates of each other.
    value.microphones[0].position_uncertainty_mm = None;
    value.geometry = CaptureGeometry::Compact;
    assert!(value.validate().is_ok());
}

#[test]
fn unknown_fields_are_not_silently_ignored() {
    let mut value = serde_json::to_value(plan()).unwrap();
    value["microphones"][0]["gain"] = serde_json::json!(12);
    assert!(serde_json::from_value::<CaptureSessionPlan>(value).is_err());
}
