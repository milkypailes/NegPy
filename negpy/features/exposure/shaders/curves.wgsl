// Display-referred curves after levels; mirrors curves.py::apply_curves.
//
// Lanes are (global, red, green, blue). The CPU bakes the same tables and both
// sides index them identically, so parity holds by construction. Identity lanes
// are skipped through the active mask: even an identity table would quantize
// to 8 bits.

struct CurvesUniforms {
    apply_mask: vec4<u32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: CurvesUniforms;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>, 256>;

fn lut_at(x: f32, lane: u32) -> f32 {
    return lut[u32(clamp(x * 255.0 + 0.5, 0.0, 255.0))][lane];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;
    if (params.apply_mask.x != 0u) {
        color = vec3<f32>(lut_at(color.r, 0u), lut_at(color.g, 0u), lut_at(color.b, 0u));
    }
    if (params.apply_mask.y != 0u) {
        color.r = lut_at(color.r, 1u);
    }
    if (params.apply_mask.z != 0u) {
        color.g = lut_at(color.g, 2u);
    }
    if (params.apply_mask.w != 0u) {
        color.b = lut_at(color.b, 3u);
    }
    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
