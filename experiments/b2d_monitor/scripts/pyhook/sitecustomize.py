"""viz_loop.py route processes only (PYTHONPATH): keep the CARLA world when the route's map is already loaded.

The leaderboard calls client.load_world(town) for every route, even when the server already has that map. For the
monitor loop, which groups routes by town, reuse the loaded world instead: destroy what a previous (possibly
cancelled) route left behind, then do exactly what LeaderboardEvaluator._load_and_wait_for_world does after its
load_world call. VIZ_REUSE_WORLD=0 turns it off. Prints one '[viz] world ...' line with the time it took.
"""
import os
import sys
import time

TARGET = 'leaderboard.leaderboard_evaluator'


def _patch(module):
    cls = module.LeaderboardEvaluator
    original = cls._load_and_wait_for_world

    def load(self, args, town):
        t0 = time.time()
        try:
            world = self.client.get_world()
            kept = world.get_map().name.split('/')[-1] == town
        except Exception:  # noqa: BLE001
            kept = False
        if not kept:
            original(self, args, town)
            print('[viz] world %s loaded in %.1f s' % (town, time.time() - t0), flush=True)
            return
        import carla
        stale = [a.id for a in world.get_actors() if a.type_id.split('.')[0] in ('vehicle', 'walker', 'sensor', 'controller', 'static')]
        if stale:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in stale])
        self.world = world
        settings = self.world.get_settings()            # from here on: the leaderboard's own body after load_world
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()
        module.CarlaDataProvider.set_client(self.client)
        module.CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        module.CarlaDataProvider.set_world(self.world)
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
        self.world.tick()
        print('[viz] world %s kept (%d stale actors destroyed) in %.1f s' % (town, len(stale), time.time() - t0), flush=True)
    cls._load_and_wait_for_world = load


if os.environ.get('VIZ_REUSE_WORLD', '1') == '1':
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is not None:
                run = spec.loader.exec_module

                def exec_module(module):
                    run(module)
                    _patch(module)
                spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Finder())
