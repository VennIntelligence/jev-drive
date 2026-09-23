"""Independent telemetry audit: raw sensors and recorded prior pose, no truth feedback."""
import math
import numpy as np

FIXED_K = .010659832
KEY = 'pose_lateral_coefficient_s2_per_m'
FIELDS = ('lateral_coefficient_s2_per_m','lateral_valid','lateral_reason','lateral_dt_s',
          'lateral_mean_speed_mps','lateral_mean_world_gyro_rps','lateral_midpoint_yaw_rad',
          'lateral_delta_xy_m')
NUMERIC_FIELDS = FIELDS[3:]


def finite(value):
    try:
        return bool(np.isfinite(np.asarray(value, float)).all()) and value is not None
    except (ValueError, TypeError):
        return False


def close(a, b, tolerance=1e-8):
    try:
        aa,bb=np.asarray(a,float),np.asarray(b,float)
        return finite(a) and finite(b) and aa.shape==bb.shape and bool(np.allclose(aa,bb,atol=tolerance,rtol=0))
    except (ValueError,TypeError):
        return False


def pose_features(row):
    out={k:np.nan for k in ('pose_position_error','raw_pose_position_error',
        'pose_heading_error_deg','pose_world_dx','pose_world_dy','pose_left_error','pose_forward_error')}
    if not (finite(row.get('pose_xy')) and finite(row.get('truth_xy')) and
            finite(row.get('truth_yaw')) and finite(row.get('pose_yaw'))):
        return out
    delta=np.asarray(row['pose_xy'],float)-np.asarray(row['truth_xy'],float)
    yaw=row['truth_yaw']
    out.update(pose_position_error=float(np.linalg.norm(delta)),
        pose_heading_error_deg=math.degrees((row['pose_yaw']-yaw+math.pi)%(2*math.pi)-math.pi),
        pose_world_dx=float(delta[0]),pose_world_dy=float(delta[1]),
        pose_left_error=float(delta@np.array([math.sin(yaw),-math.cos(yaw)])),
        pose_forward_error=float(delta@np.array([math.cos(yaw),math.sin(yaw)])))
    if finite(row.get('raw_pose_xy')):
        out['raw_pose_position_error']=float(np.linalg.norm(np.asarray(row['raw_pose_xy'])-row['truth_xy']))
    return out


def replay_sensors(records, motion, reference, k):
    """Independent full-sequence pose reconstruction, without logged pose feedback."""
    failures=[];samples=[];state=None
    try:
        scale=float(reference['gps_scale']);offset=np.asarray(reference['gps_offset'],float)
        settings=reference['adapter']
        rear=float(settings['rear_axle_offset_m']);gnss=float(settings['gnss_x_m'])
        gain=float(settings['pose_gnss_gain']);heading_gain=float(settings['pose_heading_gain'])
        if offset.shape!=(2,) or not finite([scale,*offset,rear,gnss,gain,heading_gain]):raise ValueError()
        if gain!=.05 or heading_gain!=.1:raise ValueError()
    except (KeyError,ValueError,TypeError):
        return dict(complete=False,errors=[dict(reason='missing_or_invalid_projector_settings')],frames=[])
    byframe={m.get('frame'):m for m in motion}
    wrap=lambda x:(x+math.pi)%(2*math.pi)-math.pi
    for row in records:
        failure=None;frame=row.get('frame');m=byframe.get(frame,{})
        try:
            sensors=m['sensors'];gps=np.asarray(sensors['GPS']['data'][:2],float)
            gyro=float(sensors['IMU']['data'][5]);compass=float(sensors['IMU']['data'][6])
            speed=float(sensors['SPEED']['data']['speed']);speed=0. if abs(speed)<.01 else speed
            t=float(m['sim_time']);dt=None if state is None else t-state['time']
            if not finite([*gps,gyro,speed,t]):failure='invalid_motion'
            elif state is not None and not 0<dt<=.2:failure='timestamp_discontinuity'
            elif not math.isfinite(compass):
                if state is None or state['compass_time'] is None:failure='compass_uninitialized'
                elif t-state['compass_time']>.2+1e-8:failure='compass_dropout_timeout'
            if failure is None:
                observed=wrap(compass-math.pi/2) if math.isfinite(compass) else wrap(state['yaw']+gyro*dt)
                lat,lon=np.deg2rad(gps)
                merc=np.array([6378137.*lon,-6378137.*np.log(np.tan(math.pi/4+lat/2))])
                raw=scale*merc+offset+(rear-gnss)*np.array([math.cos(observed),math.sin(observed)])
                if state is None:
                    xy=raw.copy();yaw=observed
                else:
                    middle=state['yaw']+.5*gyro*dt
                    v=.5*(speed+state['speed']);omega=.5*(gyro+state['gyro'])
                    xy=state['xy'].copy()
                    if k:
                        delta=-k*v*v*omega*dt*np.array([-np.sin(middle),np.cos(middle)])
                        if not finite(delta):failure='invalid_lateral_prediction'
                        xy+=delta
                    xy+=v*dt*np.array([math.cos(middle),math.sin(middle)])
                    yaw=wrap(state['yaw']+gyro*dt)
                    if math.isfinite(compass):yaw=wrap(yaw+heading_gain*wrap(observed-yaw))
                    xy+=gain*(raw-xy)
                if failure is None:
                    for name,expected in (('pose_xy',xy),('pose_yaw',yaw),('raw_pose_xy',raw)):
                        if not close(row.get(name),expected):failures.append(dict(frame=frame,reason='sensor_replay_mismatch:'+name))
                    samples.append(dict(frame=frame,sensor_replay_x_m=float(xy[0]),sensor_replay_y_m=float(xy[1]),
                        sensor_replay_yaw_rad=yaw,position_difference_m=float(np.linalg.norm(xy-np.asarray(row['pose_xy']))) if finite(row.get('pose_xy')) else None))
                    state=dict(time=t,xy=xy,yaw=yaw,speed=speed,gyro=gyro,
                        compass_time=t if math.isfinite(compass) else state['compass_time'])
        except (KeyError,TypeError,ValueError,IndexError,OverflowError):
            failure='malformed_motion_or_replay_input'
        if failure is not None:
            if row.get('reason')!='invalid_pose' or row.get('pose_status',{}).get('reason')!=failure:
                failures.append(dict(frame=frame,reason='sensor_fault_not_matched:'+failure))
            state=None
    return dict(complete=bool(records) and not failures,errors=failures,frames=samples)


def audit_pose(records, motion, config, variant, gain=.05, reference=None):
    k=0. if variant=='baseline-zero' else FIXED_K
    errors=[];evidence=[];previous=None
    def error(row, reason):
        errors.append(dict(frame=row.get('frame'),sim_time=row.get('sim_time'),reason=reason))
    if not close(config.get(KEY),k,1e-14):
        errors.append(dict(frame=None,reason='config_coefficient_missing_or_mismatch'))
    if config.get('aim_interpolation')!='linear':
        errors.append(dict(frame=None,reason='config_aim_must_be_explicit_linear'))
    if [r.get('frame') for r in records] != [r.get('frame') for r in motion]:
        errors.append(dict(frame=None,reason='motion_control_frame_sequence_mismatch'))
    byframe={m.get('frame'):m for m in motion}
    for row in records:
        local_start=len(errors)
        status=row.get('pose_status',{})
        missing=[name for name in FIELDS if name not in status]
        if missing:error(row,'missing_pose_status_fields:'+','.join(missing))
        if not close(status.get(FIELDS[0]),k,1e-14):error(row,'telemetry_coefficient_mismatch')
        m=byframe.get(row.get('frame'),{})
        if not close(m.get('sim_time'),row.get('sim_time')):error(row,'motion_control_time_mismatch')
        sensor=m.get('sensors',{})
        try:
            raw=float(sensor['SPEED']['data']['speed'])
            speed=0. if abs(raw)<.01 else raw
            gyro=float(sensor['IMU']['data'][5])
            gps=np.asarray(sensor['GPS']['data'][:2],float)
            for name in ('GPS','IMU','SPEED'):
                if sensor[name]['frame']!=row['frame']:error(row,'sensor_frame_mismatch:'+name)
            if not close(speed,row.get('speed_mps')):error(row,'raw_speed_vs_control_mismatch')
            if not close(gyro,-float(row.get('yaw_rate_rps',np.nan))):error(row,'raw_gyro_vs_control_mismatch')
            motion_valid=finite([raw,gyro,*gps,row.get('sim_time')])
        except (KeyError,ValueError,TypeError,IndexError):
            speed=gyro=np.nan;motion_valid=False
            error(row,'missing_or_malformed_raw_motion')
        expected_dt=expected_speed=expected_gyro=expected_mid=expected_delta=None
        interval=previous is not None and motion_valid and row.get('reason')!='invalid_pose'
        if interval:
            expected_dt=row['sim_time']-previous['time']
            if not 0<expected_dt<=.2:
                error(row,'invalid_interval_not_safe_reset');interval=False
        if interval:
            expected_speed=.5*(speed+previous['speed'])
            expected_gyro=.5*(gyro+previous['gyro'])
            expected_mid=previous['yaw']+.5*gyro*expected_dt
            right=np.array([-math.sin(expected_mid),math.cos(expected_mid)])
            expected_delta=-k*expected_speed**2*expected_gyro*expected_dt*right
            expected_reason='disabled' if k==0 else 'applied'
            if status.get('lateral_valid') is not True:error(row,'valid_interval_not_evaluated')
            if status.get('lateral_reason')!=expected_reason:error(row,'interval_reason_mismatch')
            for name, value in zip(NUMERIC_FIELDS,(expected_dt,expected_speed,expected_gyro,expected_mid,expected_delta)):
                if not close(status.get(name),value):error(row,'recompute_mismatch:'+name)
            nominal=previous['xy']+expected_speed*expected_dt*np.array([math.cos(expected_mid),math.sin(expected_mid)])
            if finite(row.get('raw_pose_xy')):
                fused=(1-gain)*(nominal+expected_delta)+gain*np.asarray(row['raw_pose_xy'])
                if not close(row.get('pose_xy'),fused):error(row,'pre_GNSS_compensation_order_or_fusion_mismatch')
            else:error(row,'missing_raw_pose_for_fusion_order')
        else:
            nominal=None
            if status.get('lateral_valid') is not False:error(row,'no_interval_valid_must_be_false')
            expected_reason=status.get('reason') if row.get('reason')=='invalid_pose' else 'no_interval'
            if status.get('lateral_reason')!=expected_reason:error(row,'no_interval_or_fault_reason_mismatch')
            if any(status.get(name) is not None for name in NUMERIC_FIELDS):error(row,'no_interval_numeric_fields_must_be_null')
        item=dict(frame=row.get('frame'),sim_time=row.get('sim_time'),expected_k=k,
            expected_dt_s=expected_dt,expected_mean_speed_mps=expected_speed,
            expected_mean_world_gyro_rps=expected_gyro,expected_midpoint_yaw_rad=expected_mid,
            expected_delta_x_m=None if expected_delta is None else float(expected_delta[0]),
            expected_delta_y_m=None if expected_delta is None else float(expected_delta[1]),
            expected_nominal_prediction_x_m=None if nominal is None else float(nominal[0]),
            expected_nominal_prediction_y_m=None if nominal is None else float(nominal[1]),
            error_count=len(errors)-local_start)
        evidence.append(item)
        previous=(dict(time=row['sim_time'],speed=speed,gyro=gyro,yaw=row['pose_yaw'],xy=np.asarray(row['pose_xy']))
                  if motion_valid and finite(row.get('pose_xy')) and finite(row.get('pose_yaw'))
                  and row.get('reason')!='invalid_pose' else None)
    replay=replay_sensors(records,motion,reference,k) if reference is not None else None
    if replay:
        errors.extend(replay['errors'])
        replay_byframe={r['frame']:r for r in replay['frames']}
        for item in evidence:item.update(replay_byframe.get(item['frame'],{}))
    return dict(complete=bool(records) and not errors,errors=errors,frames=evidence,
                sensor_replay_complete=replay['complete'] if replay else None,
                input_motion_count=len(motion),control_count=len(records))
