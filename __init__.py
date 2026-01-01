bl_info = {
    "name": "BB Cache Animation",
    "author": "Blender Bob & Claude.ai",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > N-Panel > Animation > BB Cache Animation",
    "description": "Animation caching for multiple characters",
    "category": "Animation",
}

import bpy
import struct
import numpy as np
from pathlib import Path
import shutil
import hashlib
import json
from bpy.props import (
    BoolProperty,
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    PointerProperty,
)
from bpy.types import (
    Operator,
    Panel,
    PropertyGroup,
    AddonPreferences,
)


# ============================================================================
# CACHE MANAGER - Core caching logic
# ============================================================================

class MDDCacheManager:
    """Handles MDD file writing and frame caching"""
    
    def __init__(self, obj, cache_dir, frame_start, frame_end):
        self.obj = obj
        self.cache_dir = Path(cache_dir)
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.temp_dir = self.cache_dir / ".temp" / obj.name
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track what's been baked
        self.cached_frames = set()
        self.num_verts = len(obj.data.vertices)
    
    def bake_frame(self, frame):
        """Bake a single frame to temporary storage"""
        scene = bpy.context.scene
        
        # Set frame silently
        scene.frame_current = frame
        
        # Force evaluation without viewport update
        view_layer = bpy.context.view_layer
        view_layer.update()
        
        # Get evaluated mesh
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = self.obj.evaluated_get(dg)
        mesh = obj_eval.to_mesh()
        
        try:
            # Verify topology hasn't changed
            if len(mesh.vertices) != self.num_verts:
                return False, f"Vertex count changed at frame {frame}"
            
            # Extract positions efficiently using foreach_get
            positions = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
            mesh.vertices.foreach_get('co', positions.ravel())
            
            # Save to temp file
            temp_file = self.temp_dir / f"frame_{frame:04d}.npy"
            np.save(temp_file, positions)
            
            self.cached_frames.add(frame)
            return True, None
            
        except Exception as e:
            return False, str(e)
        finally:
            # Always cleanup mesh to prevent memory leak
            obj_eval.to_mesh_clear()
    
    def stitch_mdd(self):
        """Combine all temp frames into final MDD file"""
        try:
            frames = sorted(self.cached_frames)
            if not frames:
                return False, "No frames to stitch"
            
            num_frames = len(frames)
            fps = bpy.context.scene.render.fps
            
            # Output path
            mdd_path = self.cache_dir / f"{self.obj.name}.mdd"
            
            with open(mdd_path, "wb") as f:
                # Write header (big-endian)
                f.write(struct.pack(">2i", num_frames, self.num_verts))
                
                # Write timecodes
                for i, frame in enumerate(frames):
                    timecode = frame / fps
                    f.write(struct.pack(">f", timecode))
                
                # Write all frame data
                for frame in frames:
                    frame_file = self.temp_dir / f"frame_{frame:04d}.npy"
                    positions = np.load(frame_file)
                    # MDD uses big-endian floats
                    positions.astype('>f4').tofile(f)
            
            # Save metadata
            meta = {
                "object": self.obj.name,
                "frames": list(frames),
                "num_verts": self.num_verts,
                "fps": fps,
            }
            meta_path = self.cache_dir / f"{self.obj.name}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            return True, str(mdd_path)
            
        except Exception as e:
            return False, str(e)
    
    def cleanup_temp(self):
        """Remove temporary frame files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def get_progress(self):
        """Return baking progress (0.0 to 1.0)"""
        total = self.frame_end - self.frame_start + 1
        return len(self.cached_frames) / total if total > 0 else 0.0


# ============================================================================
# DIRTY DETECTION - Track when cache needs rebuilding
# ============================================================================

class CacheDirtyTracker:
    """Detect when animation data changes"""
    
    @staticmethod
    def compute_hash(obj):
        """Generate hash of animation-relevant data"""
        data_parts = []
        
        # DON'T hash current transforms/matrices - they change during baking!
        # DON'T hash modifier visibility - that changes when toggling cache on/off!
        # Only hash the animation DATA (keyframes, actions, etc)
        
        # Hash armature's action if present
        if obj.parent and obj.parent.type == 'ARMATURE':
            armature = obj.parent
            
            # Hash armature's action
            if armature.animation_data and armature.animation_data.action:
                action = armature.animation_data.action
                data_parts.append(f"action:{action.name}:{action.id_data.name}")
                
                # Hash all fcurves - this catches keyframe changes
                if hasattr(action, 'fcurves') and action.fcurves:
                    for fcurve in action.fcurves:
                        data_parts.append(f"{fcurve.data_path}:{fcurve.array_index}")
                        # Hash keyframe count and values
                        if hasattr(fcurve, 'keyframe_points'):
                            data_parts.append(f"kf_count:{len(fcurve.keyframe_points)}")
                            # Hash first and last keyframe positions
                            if len(fcurve.keyframe_points) > 0:
                                first = fcurve.keyframe_points[0]
                                last = fcurve.keyframe_points[-1]
                                data_parts.append(f"first:{first.co[0]},{first.co[1]}")
                                data_parts.append(f"last:{last.co[0]},{last.co[1]}")
        
        # Hash which modifiers exist and their type (but NOT visibility)
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE':
                mod_obj = "None" if mod.object is None else mod.object.name
                data_parts.append(f"mod:{mod.name}:ARMATURE:{mod_obj}")
            elif mod.type != 'MESH_CACHE':  # Ignore cache modifier itself
                data_parts.append(f"mod:{mod.name}:{mod.type}")
        
        # Hash object's own keyframe data
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            data_parts.append(f"obj_action:{action.name}")
            if hasattr(action, 'fcurves') and action.fcurves:
                for fcurve in action.fcurves:
                    data_parts.append(f"{fcurve.data_path}:{len(fcurve.keyframe_points)}")
        
        # Hash shape keys if present
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                # Only hash the key setup, not current values
                data_parts.append(f"shapekey:{key.name}")
        
        # Create hash
        combined = "".join(data_parts)
        return hashlib.md5(combined.encode()).hexdigest()


# ============================================================================
# BACKGROUND WORKER - Handles idle-time baking
# ============================================================================

class BackgroundCacheWorker:
    """Global worker that bakes frames in the background"""
    
    def __init__(self):
        self.active_jobs = {}  # obj_name -> manager
        self.bake_queue = []  # [(obj_name, frame), ...]
        self.is_registered = False
        self.frame_before_baking = None  # Store frame before any baking starts
    
    def start_cache_job(self, obj):
        """Start a new caching job for an object"""
        
        # Prevent starting duplicate jobs
        if obj.name in self.active_jobs:
            return
        
        # Also check if already in queue
        if any(o == obj.name for o, f in self.bake_queue):
            return
        
        # Store original frame when first job starts
        if not self.active_jobs and self.frame_before_baking is None:
            self.frame_before_baking = bpy.context.scene.frame_current
        
        scene = bpy.context.scene
        
        # Get cache settings
        cache_settings = obj.cache_settings
        cache_dir = Path(bpy.path.abspath(cache_settings.cache_path))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manager
        manager = MDDCacheManager(
            obj,
            cache_dir,
            cache_settings.frame_start,
            cache_settings.frame_end
        )
        
        # Add to active jobs
        self.active_jobs[obj.name] = manager
        
        # Build frame queue
        frames = range(cache_settings.frame_start, cache_settings.frame_end + 1)
        for frame in frames:
            self.bake_queue.append((obj.name, frame))
        
        # Update state
        cache_settings.state = 'BAKING'
        cache_settings.progress = 0.0
        
        
        # Register timer if not already
        if not self.is_registered:
            bpy.app.timers.register(self.idle_worker, persistent=True)
            self.is_registered = True
    
    def idle_worker(self):
        """Timer callback - bakes one frame at a time"""
        
        global _auto_reenable_cache
        
        if not self.bake_queue:
            # No work to do - unregister and stop
            if self.is_registered:
                # Restore original frame when all baking is complete
                if self.frame_before_baking is not None:
                    bpy.context.scene.frame_set(self.frame_before_baking)
                    self.frame_before_baking = None
                
                self.is_registered = False
                return None  # Unregister timer
            return 0.5
        
        # Bake next frame
        obj_name, frame = self.bake_queue.pop(0)
        
        # Check if object still exists
        if obj_name not in bpy.data.objects:
            # Clean up job
            if obj_name in self.active_jobs:
                del self.active_jobs[obj_name]
            return 0.01
        
        obj = bpy.data.objects[obj_name]
        manager = self.active_jobs[obj_name]
        
        # Bake the frame
        success, error = manager.bake_frame(frame)
        
        if not success:
            # Error occurred
            obj.cache_settings.state = 'ERROR'
            obj.cache_settings.error_message = error
            del self.active_jobs[obj_name]
            # Remove remaining frames for this object
            self.bake_queue = [(o, f) for o, f in self.bake_queue if o != obj_name]
            return 0.01
        
        # Update progress
        obj.cache_settings.progress = manager.get_progress()
        
        # Check if this object's queue is complete
        remaining = [o for o, f in self.bake_queue if o == obj_name]
        if not remaining:
            # Stitch the MDD file
            success, result = manager.stitch_mdd()
            
            if success:
                # Apply mesh cache modifier
                apply_mesh_cache_modifier(obj, result)
                obj.cache_settings.state = 'READY'
                obj.cache_settings.cache_file_path = result
                
                # Check if this was an auto-rebuild that needs cache re-enabled
                if obj.name in _auto_reenable_cache:
                    obj.cache_settings.use_cached_playback = True
                    _auto_reenable_cache.discard(obj.name)
                else:
                    # AUTO-ENABLE cached playback after manual baking
                    obj.cache_settings.use_cached_playback = True
                
                # Clean up stale entries for deleted objects
                _auto_reenable_cache = {name for name in _auto_reenable_cache if name in bpy.data.objects}
                
                manager.cleanup_temp()
            else:
                obj.cache_settings.state = 'ERROR'
                obj.cache_settings.error_message = result
            
            del self.active_jobs[obj_name]
        
        # Continue quickly to next frame (very fast to minimize viewport flicker)
        return 0.001


# Global worker instance
g_worker = BackgroundCacheWorker()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_mesh_cache_modifier(obj, cache_path):
    """Add or update mesh cache modifier"""
    # Find existing cache modifier
    cache_mod = None
    for mod in obj.modifiers:
        if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
            cache_mod = mod
            break
    
    # Create new if doesn't exist
    if not cache_mod:
        cache_mod = obj.modifiers.new("AnimCache_MDD", "MESH_CACHE")
    
    # Configure modifier
    cache_mod.cache_format = 'MDD'
    cache_mod.filepath = cache_path
    cache_mod.frame_start = obj.cache_settings.frame_start
    cache_mod.forward_axis = 'POS_Y'
    cache_mod.up_axis = 'POS_Z'


def get_default_cache_path():
    """Get default cache directory path"""
    if bpy.data.filepath:
        blend_dir = Path(bpy.data.filepath).parent
        return str(blend_dir / "cache")
    else:
        return "//cache"


def bake_object_cache(obj, context):
    """
    Unified function to bake cache for an object.
    Used by all bake operators to ensure consistency.
    Returns (success, message)
    """
    if not obj or obj.type != 'MESH':
        return False, "Object must be a mesh"
    
    settings = obj.cache_settings
    scene_settings = context.scene.bb_cache_settings
    
    # Use scene settings if available, otherwise defaults
    if scene_settings.cache_directory:
        settings.cache_path = scene_settings.cache_directory
    elif not settings.cache_path:
        settings.cache_path = get_default_cache_path()
    
    # Set frame range from scene settings or timeline
    settings.frame_start = scene_settings.frame_start if scene_settings.frame_start > 0 else context.scene.frame_start
    settings.frame_end = scene_settings.frame_end if scene_settings.frame_end > 0 else context.scene.frame_end
    
    # Compute hash for dirty detection
    settings.last_hash = CacheDirtyTracker.compute_hash(obj)
    
    # Start baking job
    g_worker.start_cache_job(obj)
    
    return True, f"Started baking cache for {obj.name}"


# ============================================================================
# PROPERTY GROUP - Settings stored per object
# ============================================================================

class BB_CacheSettings(PropertyGroup):
    """Per-object cache settings"""
    
    cache_path: StringProperty(
        name="Cache Directory",
        description="Where to store cache files",
        default="",
        subtype='DIR_PATH'
    )
    
    frame_start: IntProperty(
        name="Start Frame",
        description="First frame to cache",
        default=1
    )
    
    frame_end: IntProperty(
        name="End Frame", 
        description="Last frame to cache",
        default=250
    )
    
    state: EnumProperty(
        name="Cache State",
        items=[
            ('NONE', "None", "No cache"),
            ('DIRTY', "Dirty", "Cache needs rebuilding"),
            ('BAKING', "Baking", "Currently baking cache"),
            ('READY', "Ready", "Cache ready for playback"),
            ('ERROR', "Error", "Cache error occurred"),
        ],
        default='NONE'
    )
    
    progress: FloatProperty(
        name="Progress",
        description="Baking progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    
    use_cached_playback: BoolProperty(
        name="Use Cached Playback",
        description="Play from cache instead of evaluating rig",
        default=True,
        update=lambda self, context: toggle_cached_playback(self, context)
    )
    
    cache_file_path: StringProperty(
        name="Cache File",
        description="Path to generated MDD file",
        default=""
    )
    
    error_message: StringProperty(
        name="Error Message",
        description="Last error message",
        default=""
    )
    
    last_hash: StringProperty(
        name="Last Hash",
        description="Hash of last cached state",
        default=""
    )


class BB_CacheSceneSettings(PropertyGroup):
    """Scene-level cache settings"""
    
    cache_directory: StringProperty(
        name="Cache Directory",
        description="Global cache directory for all characters",
        default="//cache",
        subtype='DIR_PATH'
    )
    
    frame_start: IntProperty(
        name="Start",
        description="Start frame for caching",
        default=1
    )
    
    frame_end: IntProperty(
        name="End",
        description="End frame for caching",
        default=250
    )


def toggle_cached_playback(settings, context):
    """Toggle between cached and live playback"""
    # Find which object these settings belong to
    obj = None
    for o in bpy.data.objects:
        if hasattr(o, 'cache_settings') and o.cache_settings == settings:
            obj = o
            break
    
    if obj is None:
        return
    
    
    # Update modifier visibility
    for mod in obj.modifiers:
        if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
            mod.show_viewport = settings.use_cached_playback
        elif mod.type == 'ARMATURE':
            mod.show_viewport = not settings.use_cached_playback
    
    # Force viewport update
    if context and hasattr(context, 'screen'):
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ============================================================================
# OPERATORS
# ============================================================================

class BB_OT_bake_cache(Operator):
    """Manually start baking cache"""
    bl_idname = "anim.bake_cache"
    bl_label = "Bake Cache"
    bl_description = "Start baking animation cache for this object"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.object is not None and 
                context.object.type == 'MESH')
    
    def execute(self, context):
        obj = context.object
        success, message = bake_object_cache(obj, context)
        
        if success:
            self.report({'INFO'}, message)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class BB_OT_clear_cache(Operator):
    """Clear cache for this object"""
    bl_idname = "anim.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "Remove cached files and reset state"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.object is not None and 
                context.object.type == 'MESH')
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        
        # Remove from active jobs
        if obj.name in g_worker.active_jobs:
            manager = g_worker.active_jobs[obj.name]
            manager.cleanup_temp()
            del g_worker.active_jobs[obj.name]
        
        # Remove from queue
        g_worker.bake_queue = [(o, f) for o, f in g_worker.bake_queue if o != obj.name]
        
        # Delete cache files
        if settings.cache_file_path:
            cache_path = Path(settings.cache_file_path)
            if cache_path.exists():
                cache_path.unlink()
            
            # Delete metadata
            meta_path = cache_path.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
        
        # Remove modifier
        for mod in obj.modifiers:
            if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
                obj.modifiers.remove(mod)
        
        # Re-enable armature
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.show_viewport = True
        
        # Reset state
        settings.state = 'NONE'
        settings.progress = 0.0
        settings.cache_file_path = ""
        settings.error_message = ""
        
        self.report({'INFO'}, f"Cleared cache for {obj.name}")
        return {'FINISHED'}


class BB_OT_stop_cache(Operator):
    """Stop currently running cache operation"""
    bl_idname = "anim.stop_cache"
    bl_label = "Stop Caching"
    bl_description = "Cancel the current cache baking operation"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and 
                obj.type == 'MESH' and
                obj.cache_settings.state == 'BAKING')
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        
        # Remove from active jobs
        if obj.name in g_worker.active_jobs:
            manager = g_worker.active_jobs[obj.name]
            manager.cleanup_temp()
            del g_worker.active_jobs[obj.name]
        
        # Remove from queue
        g_worker.bake_queue = [(o, f) for o, f in g_worker.bake_queue if o != obj.name]
        
        # Reset state
        settings.state = 'NONE'
        settings.progress = 0.0
        
        self.report({'INFO'}, f"Stopped caching for {obj.name}")
        return {'FINISHED'}


class BB_OT_set_playback_mode(Operator):
    """Set playback mode (Live Rig or Cache)"""
    bl_idname = "anim.set_playback_mode"
    bl_label = "Set Playback Mode"
    bl_description = "Switch between live rig and cached playback"
    bl_options = {'REGISTER', 'UNDO'}
    
    mode: bpy.props.EnumProperty(
        items=[
            ('LIVE', "Live Rig", "Use live rig evaluation"),
            ('CACHE', "Cache", "Use cached playback"),
        ]
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and 
                obj.type == 'MESH' and
                obj.cache_settings.state in ('READY', 'DIRTY'))
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        
        if self.mode == 'CACHE':
            settings.use_cached_playback = True
            self.report({'INFO'}, "Switched to cached playback")
        else:
            settings.use_cached_playback = False
            self.report({'INFO'}, "Switched to live rig")
        
        return {'FINISHED'}


class BB_OT_sync_frame_range(Operator):
    """Sync frame range from timeline"""
    bl_idname = "anim.sync_frame_range"
    bl_label = "Sync Frame Range"
    bl_description = "Set cache frame range to match timeline"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.object is not None and 
                context.object.type == 'MESH')
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        scene = context.scene
        
        settings.frame_start = scene.frame_start
        settings.frame_end = scene.frame_end
        
        self.report({'INFO'}, f"Synced to timeline: {scene.frame_start}-{scene.frame_end}")
        return {'FINISHED'}


class BB_OT_check_dirty(Operator):
    """Check if cache is dirty"""
    bl_idname = "anim.check_dirty"
    bl_label = "Check Dirty"
    bl_description = "Check if animation has changed since last cache"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and 
                obj.type == 'MESH' and
                obj.cache_settings.state == 'READY')
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        
        current_hash = CacheDirtyTracker.compute_hash(obj)
        
        if current_hash != settings.last_hash:
            settings.state = 'DIRTY'
            self.report({'WARNING'}, "Cache is dirty - animation has changed")
        else:
            self.report({'INFO'}, "Cache is up to date")
        
        return {'FINISHED'}


class BB_OT_force_refresh(Operator):
    """Force check for cache updates"""
    bl_idname = "anim.force_refresh"
    bl_label = "Refresh Cache Status"
    bl_description = "Manually check if cache needs updating"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'
    
    def execute(self, context):
        obj = context.object
        settings = obj.cache_settings
        
        if settings.state == 'READY':
            current_hash = CacheDirtyTracker.compute_hash(obj)
            if current_hash != settings.last_hash:
                settings.state = 'DIRTY'
                self.report({'WARNING'}, "Animation changed - cache is now dirty")
            else:
                self.report({'INFO'}, "Cache is up to date")
        
        return {'FINISHED'}


class BB_OT_bake_all(Operator):
    """Bake cache for all rigged meshes in scene"""
    bl_idname = "anim.bake_all"
    bl_label = "Bake All"
    bl_description = "Bake cache for all rigged mesh objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        baked_count = 0
        
        # Find all mesh objects with armature modifiers
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            
            # Check if it has an armature modifier
            has_armature = any(mod.type == 'ARMATURE' for mod in obj.modifiers)
            if not has_armature:
                continue
            
            # Use unified bake function
            success, message = bake_object_cache(obj, context)
            if success:
                baked_count += 1
        
        self.report({'INFO'}, f"Started baking {baked_count} objects")
        return {'FINISHED'}


class BB_OT_clear_all(Operator):
    """Clear all caches in scene"""
    bl_idname = "anim.clear_all"
    bl_label = "Clear All"
    bl_description = "Clear cache for all cached objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cleared_count = 0
        
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            
            settings = obj.cache_settings
            if settings.state in ('READY', 'BAKING', 'DIRTY', 'ERROR'):
                # Use the existing clear logic
                if obj.name in g_worker.active_jobs:
                    manager = g_worker.active_jobs[obj.name]
                    manager.cleanup_temp()
                    del g_worker.active_jobs[obj.name]
                
                g_worker.bake_queue = [(o, f) for o, f in g_worker.bake_queue if o != obj.name]
                
                if settings.cache_file_path:
                    cache_path = Path(settings.cache_file_path)
                    if cache_path.exists():
                        cache_path.unlink()
                    meta_path = cache_path.with_suffix('.json')
                    if meta_path.exists():
                        meta_path.unlink()
                
                for mod in obj.modifiers:
                    if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
                        obj.modifiers.remove(mod)
                
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.show_viewport = True
                
                settings.state = 'NONE'
                settings.progress = 0.0
                settings.cache_file_path = ""
                settings.error_message = ""
                cleared_count += 1
        
        self.report({'INFO'}, f"Cleared {cleared_count} caches")
        return {'FINISHED'}


# ============================================================================
# UI PANEL
# ============================================================================

class BB_PT_cache_panel(Panel):
    """Animation Cache panel in Object properties"""
    bl_label = "Animation Cache (MDD)"
    bl_idname = "ANIM_PT_cache_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        settings = obj.cache_settings
        
        # Force UI refresh during baking
        if settings.state == 'BAKING':
            # Tag for redraw to update progress bar
            for area in context.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()
        
        # Status indicator
        row = layout.row()
        if settings.state == 'NONE':
            row.label(text="○ Not Cached", icon='RADIOBUT_OFF')
        elif settings.state == 'DIRTY':
            row.label(text="● Dirty", icon='ERROR')
        elif settings.state == 'BAKING':
            row.label(text="● Baking", icon='TIME')
        elif settings.state == 'READY':
            row.label(text="● Ready", icon='CHECKMARK')
        elif settings.state == 'ERROR':
            row.label(text="● Error", icon='CANCEL')
        
        # Progress bar during baking
        if settings.state == 'BAKING':
            col = layout.column()
            col.prop(settings, "progress", text="Progress", slider=True)
            
            # Calculate frames done
            total_frames = settings.frame_end - settings.frame_start + 1
            frames_done = int(settings.progress * total_frames)
            col.label(text=f"{frames_done} / {total_frames} frames")
        
        # Error message
        if settings.state == 'ERROR' and settings.error_message:
            box = layout.box()
            col = box.column()
            col.alert = True
            col.label(text="Error:", icon='ERROR')
            col.label(text=settings.error_message)
        
        layout.separator()
        
        # Settings
        col = layout.column()
        col.prop(settings, "cache_path")
        
        row = col.row(align=True)
        row.prop(settings, "frame_start")
        row.prop(settings, "frame_end")
        row.operator("anim.sync_frame_range", text="", icon='TIME')
        
        layout.separator()
        
        # Actions
        row = layout.row(align=True)
        row.scale_y = 1.3
        
        if settings.state == 'BAKING':
            # Show stop button instead of bake when actively baking
            row.operator("anim.stop_cache", text="Stop", icon='CANCEL')
        else:
            row.operator("anim.bake_cache", text="Bake Cache", icon='RENDER_ANIMATION')
        
        row.operator("anim.clear_cache", text="Clear", icon='TRASH')
        
        # Playback toggle (only show if cache exists)
        if settings.state in ('READY', 'DIRTY'):
            layout.separator()
            
            # Two buttons side-by-side
            row = layout.row(align=True)
            row.scale_y = 1.5
            
            # Live Rig button
            op = row.operator("anim.set_playback_mode", text="Live Rig", icon='ARMATURE_DATA', depress=not settings.use_cached_playback)
            op.mode = 'LIVE'
            
            # Cache button
            op = row.operator("anim.set_playback_mode", text="Cache", icon='PLAY', depress=settings.use_cached_playback)
            op.mode = 'CACHE'
            
            # Dirty warning
            if settings.state == 'DIRTY':
                layout.separator()
                box = layout.box()
                box.alert = True
                box.label(text="Animation changed", icon='ERROR')
                box.operator("anim.bake_cache", text="Rebake Now", icon='FILE_REFRESH')


class BB_PT_cache_npanel(Panel):
    """BB Cache Animation N-Panel"""
    bl_label = "BB Cache Animation"
    bl_idname = "BB_PT_cache_npanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'
    
    def draw(self, context):
        layout = self.layout
        scene_settings = context.scene.bb_cache_settings
        
        # Global settings header
        box = layout.box()
        col = box.column(align=True)
        
        # Cache directory with placeholder
        col.label(text="Cache Directory:")
        col.prop(scene_settings, "cache_directory", text="", placeholder="//cache")
        
        # Frame range
        row = col.row(align=True)
        row.prop(scene_settings, "frame_start")
        row.prop(scene_settings, "frame_end")
        row.operator("anim.sync_frame_range_global", text="", icon='TIME')
        
        # Batch operations
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("anim.bake_all", text="Bake All", icon='RENDER_ANIMATION')
        row.operator("anim.clear_all", text="Clear All", icon='TRASH')
        
        layout.separator()
        
        # List all armatures and their meshes
        armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
        
        if not armatures:
            layout.label(text="No armatures in scene", icon='INFO')
            return
        
        for armature in armatures:
            # Find meshes rigged to this armature
            rigged_meshes = []
            for obj in bpy.data.objects:
                if obj.type != 'MESH':
                    continue
                
                # Check if rigged to this armature
                if obj.parent == armature:
                    rigged_meshes.append(obj)
                    continue
                
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == armature:
                        rigged_meshes.append(obj)
                        break
            
            if not rigged_meshes:
                continue
            
            # No armature header - just list the meshes directly
            for mesh_obj in rigged_meshes:
                self.draw_mesh_row(layout, mesh_obj, context)
        
        # Force UI refresh if any object is baking
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.cache_settings.state == 'BAKING':
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                break
    
    def draw_mesh_row(self, layout, obj, context):
        """Draw a single mesh row with status and buttons"""
        settings = obj.cache_settings
        
        row = layout.row(align=True)
        
        # Show progress percentage if baking, otherwise just mesh name
        if settings.state == 'BAKING':
            progress = int(settings.progress * 100)
            row.label(text=f"{obj.name} ({progress}%)")
        else:
            row.label(text=obj.name)
        
        # Buttons (icons only)
        if settings.state == 'BAKING':
            # Stop button while baking
            op = row.operator("anim.stop_cache", text="", icon='PAUSE')
            # Disable other buttons
            row.label(text="", icon='BLANK1')
            row.label(text="", icon='BLANK1')
            row.label(text="", icon='BLANK1')
        else:
            # Bake button
            op = row.operator("anim.bake_cache_single", text="", icon='FILE_REFRESH')
            op.object_name = obj.name
            
            # Clear button (only if cached)
            if settings.state in ('READY', 'DIRTY', 'ERROR'):
                op = row.operator("anim.clear_cache_single", text="", icon='TRASH')
                op.object_name = obj.name
            else:
                row.label(text="", icon='BLANK1')
            
            # Live Rig / Cache buttons (only if cached)
            if settings.state in ('READY', 'DIRTY'):
                # Live Rig button
                op = row.operator("anim.set_playback_mode_single", text="", icon='ARMATURE_DATA', 
                                 depress=not settings.use_cached_playback)
                op.object_name = obj.name
                op.mode = 'LIVE'
                
                # Cache button  
                op = row.operator("anim.set_playback_mode_single", text="", icon='PLAY',
                                 depress=settings.use_cached_playback)
                op.object_name = obj.name
                op.mode = 'CACHE'
            else:
                row.label(text="", icon='BLANK1')
                row.label(text="", icon='BLANK1')


# Single-object operators for N-panel
class BB_OT_bake_cache_single(Operator):
    """Bake cache for specific object"""
    bl_idname = "anim.bake_cache_single"
    bl_label = "Bake Cache"
    bl_options = {'REGISTER', 'UNDO'}
    
    object_name: StringProperty()
    
    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if not obj:
            return {'CANCELLED'}
        
        success, message = bake_object_cache(obj, context)
        
        if success:
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class BB_OT_clear_cache_single(Operator):
    """Clear cache for specific object"""
    bl_idname = "anim.clear_cache_single"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER', 'UNDO'}
    
    object_name: StringProperty()
    
    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if not obj:
            return {'CANCELLED'}
        
        settings = obj.cache_settings
        
        # Use existing clear logic
        if obj.name in g_worker.active_jobs:
            manager = g_worker.active_jobs[obj.name]
            manager.cleanup_temp()
            del g_worker.active_jobs[obj.name]
        
        g_worker.bake_queue = [(o, f) for o, f in g_worker.bake_queue if o != obj.name]
        
        if settings.cache_file_path:
            cache_path = Path(settings.cache_file_path)
            if cache_path.exists():
                cache_path.unlink()
            meta_path = cache_path.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
        
        for mod in obj.modifiers:
            if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
                obj.modifiers.remove(mod)
        
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.show_viewport = True
        
        settings.state = 'NONE'
        settings.progress = 0.0
        settings.cache_file_path = ""
        settings.error_message = ""
        
        return {'FINISHED'}


class BB_OT_set_playback_mode_single(Operator):
    """Set playback mode for specific object"""
    bl_idname = "anim.set_playback_mode_single"
    bl_label = "Set Playback Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    object_name: StringProperty()
    mode: EnumProperty(
        items=[
            ('LIVE', "Live Rig", ""),
            ('CACHE', "Cache", ""),
        ]
    )
    
    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if not obj:
            return {'CANCELLED'}
        
        settings = obj.cache_settings
        settings.use_cached_playback = (self.mode == 'CACHE')
        return {'FINISHED'}


class BB_OT_sync_frame_range_global(Operator):
    """Sync global frame range from timeline"""
    bl_idname = "anim.sync_frame_range_global"
    bl_label = "Sync Frame Range"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene_settings = context.scene.bb_cache_settings
        scene_settings.frame_start = context.scene.frame_start
        scene_settings.frame_end = context.scene.frame_end
        return {'FINISHED'}


# ============================================================================
# MODE CHANGE HANDLER - Auto-disable cache when editing
# ============================================================================

# Track previous mode states and cache states
_previous_modes = {}
_was_cache_enabled = {}  # Track which objects had cache enabled before pose mode
_mode_check_running = False  # Lock to prevent concurrent execution
_auto_reenable_cache = set()  # Track which objects should have cache re-enabled after rebuild
_last_mode_change_time = {}  # Track last time each armature changed mode (for debouncing)
if "_msgbus_owner" not in globals():
    _msgbus_owner = object()

def check_mode_changes():
    """Check all armatures for mode changes"""
    global _previous_modes, _was_cache_enabled, _mode_check_running, _auto_reenable_cache, _last_mode_change_time
    
    # Prevent concurrent execution
    if _mode_check_running:
        return
    
    _mode_check_running = True
    
    try:
        import time
        current_time = time.time()
        
        # Check all armatures for mode changes
        for obj in bpy.data.objects:
            if obj.type != 'ARMATURE':
                continue
            
            # Get current mode
            current_mode = obj.mode
            previous_mode = _previous_modes.get(obj.name, 'OBJECT')
            
            # Debounce - ignore if changed less than 0.5 seconds ago
            last_change = _last_mode_change_time.get(obj.name, 0)
            if current_time - last_change < 0.5:
                continue
            
            # Detect entering POSE mode
            if current_mode == 'POSE' and previous_mode != 'POSE':
                _last_mode_change_time[obj.name] = current_time
                
                # Disable cache on all rigged meshes
                for child in bpy.data.objects:
                    if child.type != 'MESH':
                        continue
                    
                    # Check if rigged to this armature
                    is_rigged = False
                    
                    # Check parent
                    if child.parent == obj:
                        is_rigged = True
                    
                    # Check armature modifier
                    if not is_rigged:
                        for mod in child.modifiers:
                            if mod.type == 'ARMATURE' and mod.object == obj:
                                is_rigged = True
                                break
                    
                    if is_rigged:
                        child_settings = child.cache_settings
                        
                        # Only process if cache exists
                        if child_settings.state in ('READY', 'DIRTY'):
                            # Remember if cache was enabled
                            was_enabled = child_settings.use_cached_playback
                            
                            if was_enabled:
                                _was_cache_enabled[child.name] = True
                                
                                # Directly disable modifiers
                                for mod in child.modifiers:
                                    if mod.type == 'MESH_CACHE' and mod.name.startswith("AnimCache_"):
                                        mod.show_viewport = False
                                    elif mod.type == 'ARMATURE':
                                        mod.show_viewport = True
                                
                                # Update the property (this will keep UI in sync)
                                child_settings.use_cached_playback = False
                                
                                # Force viewport refresh
                                for window in bpy.context.window_manager.windows:
                                    for area in window.screen.areas:
                                        if area.type == 'VIEW_3D':
                                            area.tag_redraw()
            
            # Detect exiting POSE mode
            elif previous_mode == 'POSE' and current_mode != 'POSE':
                _last_mode_change_time[obj.name] = current_time
                
                # Store current frame before starting any rebuilds
                current_frame = bpy.context.scene.frame_current
                if g_worker.frame_before_baking is None:
                    g_worker.frame_before_baking = current_frame
                
                # Check all rigged meshes
                for child in bpy.data.objects:
                    if child.type != 'MESH':
                        continue
                    
                    # Check if rigged to this armature
                    is_rigged = False
                    
                    if child.parent == obj:
                        is_rigged = True
                    
                    if not is_rigged:
                        for mod in child.modifiers:
                            if mod.type == 'ARMATURE' and mod.object == obj:
                                is_rigged = True
                                break
                    
                    if is_rigged:
                        child_settings = child.cache_settings
                        
                        # Check if cache exists (READY or DIRTY)
                        if child_settings.state in ('READY', 'DIRTY'):
                            child_settings.state = 'DIRTY'
                            
                            # Auto-rebake if cache was enabled before pose mode
                            if _was_cache_enabled.get(child.name, False):
                                # Mark for re-enabling after rebuild
                                _auto_reenable_cache.add(child.name)
                                bake_object_cache(child, bpy.context)
                                # Clear the flag
                                _was_cache_enabled.pop(child.name, None)
            
            # Update tracked mode
            _previous_modes[obj.name] = current_mode
    
    finally:
        _mode_check_running = False

@bpy.app.handlers.persistent
def dummy_handler(scene):
    """Dummy handler - not used, but kept for compatibility"""
    pass


def setup_msgbus_subscriptions():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    
    """Subscribe to object mode changes via msgbus"""
    subscribe_to = bpy.types.Object, "mode"
    
    bpy.msgbus.subscribe_rna(
        key=subscribe_to,
        owner=_msgbus_owner,
        args=tuple(),
        notify=check_mode_changes,
    )



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_orphaned_temp_files():
    """Clean up .temp directories from interrupted bakes"""
    try:
        # Check all objects for cache directories
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            
            settings = obj.cache_settings
            
            # Skip if currently baking
            if settings.state == 'BAKING':
                continue
            
            if settings.cache_path:
                cache_dir = Path(bpy.path.abspath(settings.cache_path))
                temp_dir = cache_dir / ".temp"
                
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
    except Exception as e:
        pass  # Silently ignore cleanup errors

def mode_change_poller():
    check_mode_changes()
    return 0.5  # Poll every 0.5 seconds

# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    BB_CacheSettings,
    BB_CacheSceneSettings,
    BB_OT_bake_cache,
    BB_OT_stop_cache,
    BB_OT_clear_cache,
    BB_OT_set_playback_mode,
    BB_OT_sync_frame_range,
    BB_OT_check_dirty,
    BB_OT_force_refresh,
    BB_OT_bake_all,
    BB_OT_clear_all,
    BB_OT_bake_cache_single,
    BB_OT_clear_cache_single,
    BB_OT_set_playback_mode_single,
    BB_OT_sync_frame_range_global,
    BB_PT_cache_panel,
    BB_PT_cache_npanel,
)


def register():
    # Register classes
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Add properties
    bpy.types.Object.cache_settings = PointerProperty(type=BB_CacheSettings)
    bpy.types.Scene.bb_cache_settings = PointerProperty(type=BB_CacheSceneSettings)
    
    # Clean up any leftover temp files from previous crashes/interruptions
    cleanup_orphaned_temp_files()
    
    # Register mode change poller timer
    if not bpy.app.timers.is_registered(mode_change_poller):
        bpy.app.timers.register(mode_change_poller, persistent=True)


def unregister():
    # Unregister mode change poller timer
    if bpy.app.timers.is_registered(mode_change_poller):
        bpy.app.timers.unregister(mode_change_poller)
    
    # Unregister timer if active
    if g_worker.is_registered:
        if bpy.app.timers.is_registered(g_worker.idle_worker):
            bpy.app.timers.unregister(g_worker.idle_worker)
    
    # Remove properties
    del bpy.types.Object.cache_settings
    del bpy.types.Scene.bb_cache_settings
    
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
