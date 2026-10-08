"""Logging gRPC tap between the AlpaSim runtime and an external driver.

Serves egodriver.EgodriverService on --listen, forwards every call unchanged to --upstream and appends one JSON
line per call to --out (wall time, method, session, upstream latency, and the payload facts a driver author
needs: camera ids / image size / timestamps, ego poses and dynamic states, route waypoints, Drive timing).
Run with the AlpaSim env's python (needs alpasim_grpc). Their driver code is not touched.
"""
import argparse, io, json, threading, time
from concurrent import futures

import grpc
from alpasim_grpc.v0 import egodriver_pb2
from google.protobuf import message_factory
from google.protobuf.json_format import MessageToDict

SERVICE = egodriver_pb2.DESCRIPTOR.services_by_name["EgodriverService"]
OPTS = [("grpc.max_send_message_length", 256 << 20), ("grpc.max_receive_message_length", 256 << 20)]


def image_facts(buf: bytes) -> dict:
    out = {"bytes": len(buf), "magic": buf[:4].hex()}
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(buf))
        out.update(format=im.format, width=im.width, height=im.height, mode=im.mode)
    except Exception as e:  # raw tensors or missing PIL: keep the byte count only
        out["decode_error"] = repr(e)[:80]
    return out


def facts(name: str, req, resp) -> dict:
    d = lambda m: MessageToDict(m, preserving_proto_field_name=True)
    if name == "start_session":
        return {"request": d(req)}
    if name == "submit_image_observation":
        c = req.camera_image
        return {"logical_id": c.logical_id, "frame_start_us": c.frame_start_us, "frame_end_us": c.frame_end_us,
                **image_facts(c.image_bytes)}
    if name == "submit_egomotion_observation":
        p, s = req.trajectory.poses, req.dynamic_states
        return {"n_poses": len(p), "n_states": len(s), "ts_us": [x.timestamp_us for x in p],
                "last_pose": d(p[-1]) if p else None, "last_state": d(s[-1]) if s else None}
    if name == "submit_route":
        w = req.route.waypoints
        return {"timestamp_us": req.route.timestamp_us, "n_waypoints": len(w), "waypoints": [[x.x, x.y, x.z] for x in w]}
    if name == "submit_recording_ground_truth":
        return {"timestamp_us": req.ground_truth.timestamp_us, "n_poses": len(req.ground_truth.trajectory.poses)}
    if name == "drive":
        p = resp.trajectory.poses if resp is not None else []
        return {"time_now_us": req.time_now_us, "time_query_us": req.time_query_us,
                "renderer_data_bytes": len(req.renderer_data), "resp_n_poses": len(p),
                "resp_ts_us": [x.timestamp_us for x in p], "resp_first_pose": d(p[0]) if p else None,
                "resp_last_pose": d(p[-1]) if p else None,
                "terminate_session": getattr(resp, "terminate_session", None)}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", required=True, help="host:port to serve on")
    ap.add_argument("--upstream", required=True, help="host:port of the real driver")
    ap.add_argument("--out", required=True, help="JSONL file")
    a = ap.parse_args()
    chan = grpc.insecure_channel(a.upstream, options=OPTS)
    log, lock = open(a.out, "a", buffering=1), threading.Lock()

    def handler(m):
        req_cls, resp_cls = message_factory.GetMessageClass(m.input_type), message_factory.GetMessageClass(m.output_type)
        call = chan.unary_unary(f"/{SERVICE.full_name}/{m.name}", request_serializer=req_cls.SerializeToString,
                                response_deserializer=resp_cls.FromString)

        def fn(req, ctx):
            t0, resp, err, left = time.time(), None, None, ctx.time_remaining()
            try:
                resp = call(req)  # the caller's deadline is logged, not forwarded
                return resp
            except grpc.RpcError as e:
                err = f"{e.code().name}: {e.details()}"
                ctx.abort(e.code(), e.details())
            finally:
                rec = {"t": t0, "method": m.name, "session": getattr(req, "session_uuid", None),
                       "ms": round((time.time() - t0) * 1e3, 3), "error": err, "req_bytes": req.ByteSize(),
                       "deadline_s": left}
                try:
                    rec.update(facts(m.name, req, resp))
                except Exception as e:
                    rec["facts_error"] = repr(e)[:200]
                with lock:
                    log.write(json.dumps(rec) + "\n")

        return grpc.unary_unary_rpc_method_handler(fn, request_deserializer=req_cls.FromString,
                                                   response_serializer=resp_cls.SerializeToString)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32), options=OPTS)
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(SERVICE.full_name, {m.name: handler(m) for m in SERVICE.methods}),))
    server.add_insecure_port(a.listen)
    server.start()
    print(f"tap {a.listen} -> {a.upstream}, log {a.out}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
