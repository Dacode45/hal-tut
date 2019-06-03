#![allow(clippy::len_zero)]
#![allow(clippy::many_single_char_names)]

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

use arrayvec::ArrayVec;
use core::mem::{size_of, ManuallyDrop};
use gfx_hal::{
        adapter::{Adapter, MemoryTypeId, PhysicalDevice},
        buffer::Usage as BufferUsage,
        command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
        device::Device,
        format::{Aspects, ChannelType, Format, Swizzle},
        image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
        memory::{Properties, Requirements},
        pass::{
                Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass,
                SubpassDesc,
        },
        pool::{CommandPool, CommandPoolCreateFlags},
        pso::{
                AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState,
                ColorBlendDesc, ColorMask, DepthStencilDesc, DepthTest, DescriptorSetLayoutBinding,
                Element, EntryPoint, Face, Factor, FrontFace, GraphicsPipelineDesc,
                GraphicsShaderSet, InputAssemblerDesc, LogicOp, PipelineCreationFlags,
                PipelineStage, PolygonMode, Rasterizer, Rect, ShaderStageFlags, Specialization,
                StencilTest, VertexBufferDesc, Viewport,
        },
        queue::{family::QueueGroup, Submission},
        window::{Backbuffer, Extent2D, FrameSync, PresentMode, Swapchain, SwapchainConfig},
        Backend, Gpu, Graphics, Instance, Primitive, QueueFamily, Surface,
};
use winit::{
        dpi::LogicalSize, CreationError, Event, EventsLoop, Window, WindowBuilder, WindowEvent,
};

pub const WINDOW_NAME: &str = "Triangle Intro";

pub const VERTEX_SOURCE: &str = "#version 450
layout (location = 0) in vec2 position;

out gl_PerVertex {
  vec4 gl_Position;
};

void main()
{
  gl_Position = vec4(position, 0.0, 1.0);
}";

pub const FRAGMENT_SOURCE: &str = "#version 450
layout(location = 0) out vec4 color;

void main()
{
  color = vec4(1.0);
}";

#[derive(Debug, Clone, Copy)]
/// Describes a triangle
pub struct Triangle {
        /// We use a 2D array of xy points.
        pub points: [[f32; 2]; 3],
}

impl Triangle {
        /// We flatten the points into a single buffer of 6 points for buffers
        pub fn points_flat(self) -> [f32; 6] {
                let [[a, b], [c, d], [e, f]] = self.points;
                [a, b, c, d, e, f]
        }
}

/// HalState controls the entire cpu-gpu pipeline
pub struct HalState {
        buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
        memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
        descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
        pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
        graphics_pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
        requirements: Requirements,
        // index into the swapchain to use
        current_frame: usize,
        // frames_in_flight is the number of images in the swap_chain
        // 3 for triplebuffering 2 otherwise
        frames_in_flight: usize,
        // fence to prevent writes to an image before it is done displaying
        in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
        // This semaphore allows us to pause the swapchain present until the rendering is finished.
        /// The wait is done by creating a Submission {} with a signal_semaphore using render finished
        render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
        /// Semaphore waits on the swapchain for the image to be ready to render
        image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
        /// Command buffers for eqch frame
        command_buffers: Vec<CommandBuffer<back::Backend, Graphics, MultiShot, Primary>>,
        // Command pool is a memory allocator for command buffers
        command_pool: ManuallyDrop<CommandPool<back::Backend, Graphics>>,
        // Framebuffers for the image view. Basically a region of memory. Configured by a render pass.
        framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
        // Only one, we are using a typical 2D image view with just color info
        image_views: Vec<(<back::Backend as Backend>::ImageView)>,
        // Passed to pipeline and command buffer. Determines what ops are used. What format, layouts, depth, stencil, color attachement etc.
        render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
        // What part of the extent to render to
        render_area: Rect,
        // queue_group we use for submitting graphics commands.
        queue_group: QueueGroup<back::Backend, Graphics>,
        // swapchain is the tripple buffered images we used.
        swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
        // device is the logical device. Dropped before the adapter
        device: ManuallyDrop<back::Device>,
        // An adapter is something that supports the usage of the api we are using.
        // Usually this is a GPU but it could be a software implementation.
        // This is equivalent to a physical device in vulkan
        _adapter: Adapter<back::Backend>,
        // surface is our winit
        _surface: <back::Backend as Backend>::Surface,
        // This is the backend
        _instance: ManuallyDrop<back::Instance>,
}
impl HalState {
        /// Creates a new, fully initialized HalState.
        pub fn new(window: &Window) -> Result<Self, &'static str> {
                // Create An Instance
                let instance = back::Instance::create(WINDOW_NAME, 1);

                // Create A Surface
                let mut surface = instance.create_surface(window);

                // Select An Adapter
                let adapter = instance
                        // go through adapters available on the machine
                        .enumerate_adapters()
                        .into_iter()
                        // find the first that supports graphics and queue family
                        .find(|a| {
                                a.queue_families.iter().any(|qf| {
                                        qf.supports_graphics() && surface.supports_queue_family(qf)
                                })
                        })
                        .ok_or("Couldn't find a graphical Adapter!")?;

                // Open A Device and take out a QueueGroup from a queue family that supports submitting graphics command
                // The device is the logical device we use not the physical device(adapter)
                // the Queue_group is the first queue_group of the adapter queue_family that supports graphics commands
                let (mut device, queue_group) = {
                        // Regain the queue_family
                        let queue_family = adapter
                                .queue_families
                                .iter()
                                // find the first q family that supports graphics and the surface
                                .find(|qf| {
                                        qf.supports_graphics() && surface.supports_queue_family(qf)
                                })
                                .ok_or("Couldn't find a QueueFamily with graphics!")?;
                        let Gpu { device, mut queues } = unsafe {
                                // grab the phsyical device from the adapter and open with the
                                // given queue family.
                                adapter.physical_device
                                        // Create a logical device by passing the priority we want.
                                        // We can use multiple devices and assign floating point priorities
                                        // between 0.0 and 1.0. Here we use 1 device so make scheduling easy.
                                        .open(&[(&queue_family, &[1.0; 1])])
                                        .map_err(|_| "Couldn't open the PhysicalDevice!")?
                        };
                        // Get the queuue with grpahics support
                        let queue_group = queues
                                .take::<Graphics>(queue_family.id())
                                .ok_or("Couldn't take ownership of the QueueGroup!")?;
                        if queue_group.queues.len() > 0 {
                                Ok(())
                        } else {
                                Err("The QueueGroup did not have any CommandQueues available!")
                        }?;
                        (device, queue_group)
                };

                // Create A Swapchain, this is extra long
                // swapchain - typical tripple buffered swapchain
                // extent - 2D extent of the window
                // backbuffer -
                // format - color format
                // frames_in_flight
                let (swapchain, extent, backbuffer, format, frames_in_flight) = {
                        // caps - surface capabilities
                        // preferred_formats - colors expected for the surface
                        // present_mode - vsynced or not. fifo for first come first served
                        // composite_alpha - can we interact with other windows.
                        let (caps, preferred_formats, present_modes, composite_alphas) =
                                surface.compatibility(&adapter.physical_device);
                        info!("{:?}", caps);
                        info!("Preferred Formats: {:?}", preferred_formats);
                        info!("Present Modes: {:?}", present_modes);
                        info!("Composite Alphas: {:?}", composite_alphas);

                        let present_mode = {
                                use gfx_hal::window::PresentMode::*;
                                [Mailbox, Fifo, Relaxed, Immediate]
                                        .iter()
                                        .cloned()
                                        .find(|pm| present_modes.contains(pm))
                                        .ok_or("No PresentMode values specified!")?
                        };
                        let composite_alpha = {
                                use gfx_hal::window::CompositeAlpha::*;
                                [Opaque, Inherit, PreMultiplied, PostMultiplied]
                                        .iter()
                                        .cloned()
                                        .find(|ca| composite_alphas.contains(ca))
                                        .ok_or("No CompositeAlpha values specified!")?
                        };
                        let format = match preferred_formats {
                                None => Format::Rgba8Srgb,
                                Some(formats) => match formats
                                        .iter()
                                        .find(|format| format.base_format().1 == ChannelType::Srgb)
                                        .cloned()
                                {
                                        Some(srgb_format) => srgb_format,
                                        None => formats
                                                .get(0)
                                                .cloned()
                                                .ok_or("Preferred format list was empty!")?,
                                },
                        };
                        let extent = {
                                let window_client_area = window
                                        .get_inner_size()
                                        .ok_or("Window doesn't exist!")?
                                        .to_physical(window.get_hidpi_factor());
                                Extent2D {
                                        width: caps
                                                .extents
                                                .end
                                                .width
                                                .min(window_client_area.width as u32),
                                        height: caps
                                                .extents
                                                .end
                                                .height
                                                .min(window_client_area.height as u32),
                                }
                        };
                        // Decide whether to triple or double buffer here
                        let image_count = if present_mode == PresentMode::Mailbox {
                                (caps.image_count.end - 1).min(3)
                        } else {
                                (caps.image_count.end - 1).min(2)
                        };
                        let image_layers = 1;
                        // Usage determines where we can use the image data
                        //https://docs.rs/gfx-hal/0.1.0/gfx_hal/image/struct.Usage.html
                        let image_usage = if caps.usage.contains(Usage::COLOR_ATTACHMENT) {
                                Usage::COLOR_ATTACHMENT
                        } else {
                                Err("The Surface isn't capable of supporting color!")?
                        };
                        let swapchain_config = SwapchainConfig {
                                present_mode,
                                composite_alpha,
                                format,
                                extent,
                                image_count,
                                image_layers,
                                image_usage,
                        };
                        info!("{:?}", swapchain_config);
                        // create a swapchain on the surface.
                        // Since we aren't destroying an old one pass in none instead of
                        // Some(old_swapchain)
                        let (swapchain, backbuffer) = unsafe {
                                device.create_swapchain(&mut surface, swapchain_config, None)
                                        .map_err(|_| "Failed to create the swapchain!")?
                        };
                        (swapchain, extent, backbuffer, format, image_count as usize)
                };

                // Create Our Sync Primitives
                // We need 1 set of semaphores for every image in the swap chain
                // Grab an image index that signals image_available semaphore once its fully ready
                // Get sync primitives out of ring buffer
                // wait on flight_fence for the image index to know we are clare to use the position
                // reset fence so we can pass it as a part of submission later
                // record our command buffer while we wait
                // submit a command buffer to wait on image available and signal render finish and flight fence
                // present swapchain and wait on render finish
                let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = {
                        let mut image_available_semaphores: Vec<
                                <back::Backend as Backend>::Semaphore,
                        > = vec![];
                        let mut render_finished_semaphores: Vec<
                                <back::Backend as Backend>::Semaphore,
                        > = vec![];
                        let mut in_flight_fences: Vec<<back::Backend as Backend>::Fence> = vec![];
                        for _ in 0..frames_in_flight {
                                in_flight_fences.push(device
                                        .create_fence(true)
                                        .map_err(|_| "Could not create a fence!")?);
                                image_available_semaphores.push(device
                                        .create_semaphore()
                                        .map_err(|_| "Could not create a semaphore!")?);
                                render_finished_semaphores.push(device
                                        .create_semaphore()
                                        .map_err(|_| "Could not create a semaphore!")?);
                        }
                        (
                                image_available_semaphores,
                                render_finished_semaphores,
                                in_flight_fences,
                        )
                };

                // Define A RenderPass
                let render_pass = {
                        let color_attachment = Attachment {
                                format: Some(format),
                                samples: 1,
                                ops: AttachmentOps {
                                        load: AttachmentLoadOp::Clear,
                                        store: AttachmentStoreOp::Store,
                                },
                                stencil_ops: AttachmentOps::DONT_CARE,
                                layouts: Layout::Undefined..Layout::Present,
                        };
                        let subpass = SubpassDesc {
                                colors: &[(0, Layout::ColorAttachmentOptimal)],
                                depth_stencil: None,
                                inputs: &[],
                                resolves: &[],
                                preserves: &[],
                        };
                        unsafe {
                                device.create_render_pass(&[color_attachment], &[subpass], &[])
                                        .map_err(|_| "Couldn't create a render pass!")?
                        }
                };

                // Create The ImageViews
                // We get them from the backbuffer assuming it is just a set of images
                let image_views: Vec<_> = match backbuffer {
                        Backbuffer::Images(images) => {
                                images.into_iter()
                                        .map(|image| unsafe {
                                                device
            .create_image_view(
              &image,
              // 2 Dimmensional array
              ViewKind::D2,
              // color format
              format,
              // don't need to remap xyz to something rediculus like xzy
              Swizzle::NO,
              SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
              },
            )
            .map_err(|_| "Couldn't create the image_view for the image!")
                                        })
                                        .collect::<Result<Vec<_>, &str>>()?
                        }
                        Backbuffer::Framebuffer(_) => {
                                unimplemented!("Can't handle framebuffer backbuffer!")
                        }
                };

                // Create Our FrameBuffers
                // Every image in our backbuffer 3 in the case of tripple buffering
                // Gets a frame buffer to actually render things to
                let framebuffers: Vec<<back::Backend as Backend>::Framebuffer> = {
                        image_views
                                .iter()
                                .map(|image_view| unsafe {
                                        device.create_framebuffer(
                                                &render_pass,
                                                vec![image_view],
                                                Extent {
                                                        width: extent.width as u32,
                                                        height: extent.height as u32,
                                                        depth: 1,
                                                },
                                        )
                                        .map_err(|_| "Failed to create a framebuffer!")
                                })
                                .collect::<Result<Vec<_>, &str>>()?
                };

                // Create Our CommandPool
                let mut command_pool = unsafe {
                        device.create_command_pool_typed(
                                &queue_group,
                                // Not transient (short lived)
                                CommandPoolCreateFlags::RESET_INDIVIDUAL,
                        )
                        .map_err(|_| "Could not create the raw command pool!")?
                };

                // Create Our CommandBuffers
                let command_buffers: Vec<_> = framebuffers
                        .iter()
                        // We use our command pool to acquire memory for our command buffers
                        .map(|_| command_pool.acquire_command_buffer())
                        .collect();

                // Build our pipeline and vertex buffer
                let (descriptor_set_layouts, pipeline_layout, graphics_pipeline) =
                        Self::create_pipeline(&mut device, extent, &render_pass)?;
                let (buffer, memory, requirements) = unsafe {
                        // Get the exact size of our triangle struct when it is flattened 2D coordinates * 3 coordinates
                        const F32_XY_TRIANGLE: u64 = (size_of::<f32>() * 2 * 3) as u64;
                        let mut buffer = device
                                .create_buffer(F32_XY_TRIANGLE, BufferUsage::VERTEX)
                                .map_err(|_| "Couldn't create a buffer for the verticies")?;
                        let mut requirements = device.get_buffer_requirements(&buffer);
                        let memory_type_id = adapter
                                .physical_device
                                // Get memory properties (visability) cpu visible, gpu visible, etc
                                .memory_properties()
                                .memory_types
                                .iter()
                                .enumerate()
                                // find memory that is cpu visible
                                .find(|&(id, memory_type)| {
                                        requirements.type_mask & (1 << id) != 0
                                                && memory_type
                                                        .properties
                                                        .contains(Properties::CPU_VISIBLE)
                                })
                                .map(|(id, _)| MemoryTypeId(id))
                                .ok_or(
                                        "Couldn't find a memory type to support the vertex buffer!",
                                )?;
                        let memory = device
                                .allocate_memory(memory_type_id, requirements.size)
                                .map_err(|_| "Couldn't allocate vertex buffer memory")?;
                        device.bind_buffer_memory(&memory, 0, &mut buffer)
                                .map_err(|_| "Couldn't bind the buffer memory!")?;
                        (buffer, memory, requirements)
                };

                Ok(Self {
                        requirements,
                        buffer: ManuallyDrop::new(buffer),
                        memory: ManuallyDrop::new(memory),
                        _instance: ManuallyDrop::new(instance),
                        _surface: surface,
                        _adapter: adapter,
                        device: ManuallyDrop::new(device),
                        queue_group,
                        swapchain: ManuallyDrop::new(swapchain),
                        render_area: extent.to_extent().rect(),
                        render_pass: ManuallyDrop::new(render_pass),
                        image_views,
                        framebuffers,
                        command_pool: ManuallyDrop::new(command_pool),
                        command_buffers,
                        image_available_semaphores,
                        render_finished_semaphores,
                        in_flight_fences,
                        frames_in_flight,
                        current_frame: 0,
                        descriptor_set_layouts,
                        pipeline_layout: ManuallyDrop::new(pipeline_layout),
                        graphics_pipeline: ManuallyDrop::new(graphics_pipeline),
                })
        }

        #[allow(clippy::type_complexity)]
        // creates a new rendering pipeline for use in draw and clear functions
        pub fn create_pipeline(
                device: &mut back::Device,
                extent: Extent2D,
                render_pass: &<back::Backend as Backend>::RenderPass,
        ) -> Result<
                (
                        Vec<<back::Backend as Backend>::DescriptorSetLayout>,
                        <back::Backend as Backend>::PipelineLayout,
                        <back::Backend as Backend>::GraphicsPipeline,
                ),
                &'static str,
        > {
                let mut compiler = shaderc::Compiler::new().ok_or("shaderc not found!")?;
                let vertex_compile_artifact = compiler
                        .compile_into_spirv(
                                VERTEX_SOURCE,
                                shaderc::ShaderKind::Vertex,
                                "vertex.vert",
                                "main",
                                None,
                        )
                        .map_err(|_| "Couldn't compile vertex shader!")?;
                let fragment_compile_artifact = compiler
                        .compile_into_spirv(
                                FRAGMENT_SOURCE,
                                shaderc::ShaderKind::Fragment,
                                "fragment.fragment",
                                "main",
                                None,
                        )
                        .map_err(|e| {
                                error!("{}", e);
                                "Couldn't compile fragment shader!"
                        })?;
                let vertex_shader_module = unsafe {
                        device.create_shader_module(vertex_compile_artifact.as_binary_u8())
                                .map_err(|_| "Couldn't make the vertex module")?
                };
                let fragment_shader_module = unsafe {
                        device.create_shader_module(fragment_compile_artifact.as_binary_u8())
                                .map_err(|_| "Couldn't make the fragment module")?
                };
                let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) = {
                        let (vs_entry, fs_entry) = (
                                EntryPoint {
                                        entry: "main",
                                        module: &vertex_shader_module,
                                        specialization: Specialization {
                                                constants: &[],
                                                data: &[],
                                        },
                                },
                                EntryPoint {
                                        entry: "main",
                                        module: &fragment_shader_module,
                                        specialization: Specialization {
                                                constants: &[],
                                                data: &[],
                                        },
                                },
                        );

                        let shaders = GraphicsShaderSet {
                                vertex: vs_entry,
                                hull: None,
                                domain: None,
                                geometry: None,
                                fragment: Some(fs_entry),
                        };

                        let input_assembler = InputAssemblerDesc::new(Primitive::TriangleList);

                        let vertex_buffers: Vec<VertexBufferDesc> = vec![VertexBufferDesc {
                                binding: 0,
                                // 2 coords per vertex
                                stride: (size_of::<f32>() * 2) as u32,
                                rate: 0,
                        }];
                        let attributes: Vec<AttributeDesc> = vec![AttributeDesc {
                                location: 0,
                                binding: 0,
                                element: Element {
                                        format: Format::Rg32Float,
                                        offset: 0,
                                },
                        }];

                        let rasterizer = Rasterizer {
                                depth_clamping: false,
                                // how the rasterizer should work
                                polygon_mode: PolygonMode::Fill,
                                // Backface calling
                                cull_face: Face::NONE,
                                // what direction are verticies in
                                front_face: FrontFace::Clockwise,
                                //
                                depth_bias: None,
                                conservative: false,
                        };

                        let depth_stencil = DepthStencilDesc {
                                // don't need depth testing right now
                                depth: DepthTest::Off,
                                depth_bounds: false,
                                stencil: StencilTest::Off,
                        };

                        let blender = {
                                let blend_state = BlendState::On {
                                        color: BlendOp::Add {
                                                src: Factor::One,
                                                dst: Factor::Zero,
                                        },
                                        alpha: BlendOp::Add {
                                                src: Factor::One,
                                                dst: Factor::Zero,
                                        },
                                };
                                BlendDesc {
                                        logic_op: Some(LogicOp::Copy),
                                        targets: vec![ColorBlendDesc(ColorMask::ALL, blend_state)],
                                }
                        };

                        let baked_states = BakedStates {
                                viewport: Some(Viewport {
                                        rect: extent.to_extent().rect(),
                                        depth: (0.0..1.0),
                                }),
                                scissor: Some(extent.to_extent().rect()),
                                blend_color: None,
                                depth_bounds: None,
                        };

                        let bindings = Vec::<DescriptorSetLayoutBinding>::new();
                        let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
                        let descriptor_set_layouts: Vec<
                                <back::Backend as Backend>::DescriptorSetLayout,
                        > = vec![unsafe {
                                device.create_descriptor_set_layout(bindings, immutable_samplers)
                                        .map_err(|_| "Couldn't make a DescriptorSetLayout")?
                        }];
                        let push_constants =
                                Vec::<(ShaderStageFlags, core::ops::Range<u32>)>::new();
                        let layout = unsafe {
                                device.create_pipeline_layout(
                                        &descriptor_set_layouts,
                                        push_constants,
                                )
                                .map_err(|_| "Couldn't create a pipeline layout")?
                        };

                        let gfx_pipeline = {
                                let desc = GraphicsPipelineDesc {
                                        shaders,
                                        rasterizer,
                                        vertex_buffers,
                                        attributes,
                                        input_assembler,
                                        blender,
                                        depth_stencil,
                                        multisampling: None,
                                        baked_states,
                                        layout: &layout,
                                        subpass: Subpass {
                                                index: 0,
                                                main_pass: render_pass,
                                        },
                                        flags: PipelineCreationFlags::empty(),
                                        parent: BasePipeline::None,
                                };

                                unsafe {
                                        device.create_graphics_pipeline(&desc, None).map_err(
                                                |_| "Couldn't create a graphics pipeline!",
                                        )?
                                }
                        };

                        (descriptor_set_layouts, layout, gfx_pipeline)
                };

                unsafe {
                        // once it's in the pipline get rid of it
                        device.destroy_shader_module(vertex_shader_module);
                        device.destroy_shader_module(fragment_shader_module);
                }

                Ok((descriptor_set_layouts, pipeline_layout, gfx_pipeline))
        }

        /// Draw a frame that's just cleared to the color specified.
        pub fn draw_clear_frame(&mut self, color: [f32; 4]) -> Result<(), &'static str> {
                // SETUP FOR THIS FRAME
                let image_available = &self.image_available_semaphores[self.current_frame];
                let render_finished = &self.render_finished_semaphores[self.current_frame];
                // Advance the frame _before_ we start using the `?` operator
                self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

                let (i_u32, i_usize) = unsafe {
                        let image_index = self
                                .swapchain
                                .acquire_image(
                                        core::u64::MAX,
                                        FrameSync::Semaphore(image_available),
                                )
                                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
                        (image_index, image_index as usize)
                };

                let flight_fence = &self.in_flight_fences[i_usize];
                unsafe {
                        self.device
                                .wait_for_fence(flight_fence, core::u64::MAX)
                                .map_err(|_| "Failed to wait on the fence!")?;
                        self.device
                                .reset_fence(flight_fence)
                                .map_err(|_| "Couldn't reset the fence!")?;
                }

                // RECORD COMMANDS
                unsafe {
                        let buffer = &mut self.command_buffers[i_usize];
                        let clear_values = [ClearValue::Color(ClearColor::Float(color))];
                        buffer.begin(false);
                        buffer.begin_render_pass_inline(
                                &self.render_pass,
                                &self.framebuffers[i_usize],
                                self.render_area,
                                clear_values.iter(),
                        );
                        buffer.finish();
                }

                // SUBMISSION AND PRESENT
                let command_buffers = &self.command_buffers[i_usize..=i_usize];
                let wait_semaphores: ArrayVec<[_; 1]> =
                        [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
                let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
                // yes, you have to write it twice like this. yes, it's silly.
                let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
                let submission = Submission {
                        command_buffers,
                        wait_semaphores,
                        signal_semaphores,
                };
                let the_command_queue = &mut self.queue_group.queues[0];
                unsafe {
                        the_command_queue.submit(submission, Some(flight_fence));
                        self.swapchain
                                .present(the_command_queue, i_u32, present_wait_semaphores)
                                .map_err(|_| "Failed to present into the swapchain!")
                }
        }

        /// Draw a frame that's just cleared to the color specified.
        pub fn draw_triangle_frame(&mut self, triangle: Triangle) -> Result<(), &'static str> {
                // SETUP FOR THIS FRAME
                let image_available = &self.image_available_semaphores[self.current_frame];
                let render_finished = &self.render_finished_semaphores[self.current_frame];
                // Advance the frame before using the `?` operator
                self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

                // Get the index to the image in the swap chain
                let (i_u32, i_usize) = unsafe {
                        let image_index = self
                                .swapchain
                                .acquire_image(
                                        core::u64::MAX,
                                        FrameSync::Semaphore(image_available),
                                )
                                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
                        (image_index, image_index as usize)
                };

                // Remember that the swap chain doesn't prevent us from writing to multiple images at once.
                // The image we want to present could be being written to so we have to wait.
                // It's nice that we don't have to reset a fence the moment it is done (fences have downtime)
                let flight_fence = &self.in_flight_fences[i_usize];
                unsafe {
                        self.device
                                .wait_for_fence(flight_fence, core::u64::MAX)
                                .map_err(|_| "Failed to wait on fence!")?;
                        self.device
                                .reset_fence(flight_fence)
                                .map_err(|_| "Couldn't reset the fence!")?;
                }

                // Write the Triangle Data
                unsafe {
                        let mut data_target = self
                                .device
                                .acquire_mapping_writer(&self.memory, 0..self.requirements.size)
                                .map_err(|_| "Failed to acquire a memory writer!")?;
                        let points = triangle.points_flat();
                        data_target[..points.len()].copy_from_slice(&points);
                        self.device
                                .release_mapping_writer(data_target)
                                .map_err(|_| "Couldn't release the mapping writer!")?;
                }

                // RECORD COMMANDS
                unsafe {
                        let buffer = &mut self.command_buffers[i_usize];
                        const TRIANGLE_CLEAR: [ClearValue; 1] =
                                [ClearValue::Color(ClearColor::Float([0.1, 0.2, 0.3, 1.0]))];
                        buffer.begin(false);
                        {
                                let mut encoder = buffer.begin_render_pass_inline(
                                        &self.render_pass,
                                        &self.framebuffers[i_usize],
                                        self.render_area,
                                        TRIANGLE_CLEAR.iter(),
                                );
                                encoder.bind_graphics_pipeline(&self.graphics_pipeline);
                                let buffer_ref: &<back::Backend as Backend>::Buffer = &self.buffer;
                                let buffers: ArrayVec<[_; 1]> = [(buffer_ref, 0)].into();
                                encoder.bind_vertex_buffers(0, buffers);
                                encoder.draw(0..3, 0..1);
                        }
                        buffer.finish();
                }

                // SUBMISSION AND PRESENT
                let command_buffers = &self.command_buffers[i_usize..=i_usize];
                let wait_semaphores: ArrayVec<[_; 1]> =
                        [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
                let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
                let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
                let submission = Submission {
                        command_buffers,
                        wait_semaphores,
                        signal_semaphores,
                };

                let the_command_queue = &mut self.queue_group.queues[0];
                unsafe {
                        the_command_queue.submit(submission, Some(flight_fence));
                        self.swapchain
                                .present(the_command_queue, i_u32, present_wait_semaphores)
                                .map_err(|_| "Failed to present into the swapchain!")
                }
        }
}
impl core::ops::Drop for HalState {
        /// We have to clean up "leaf" elements before "root" elements. Basically, we
        /// clean up in reverse of the order that we created things.
        fn drop(&mut self) {
                let _ = self.device.wait_idle();
                unsafe {
                        for descriptor_set_layout in self.descriptor_set_layouts.drain(..) {
                                self.device
                                        .destroy_descriptor_set_layout(descriptor_set_layout)
                        }
                        for fence in self.in_flight_fences.drain(..) {
                                self.device.destroy_fence(fence)
                        }
                        for semaphore in self.render_finished_semaphores.drain(..) {
                                self.device.destroy_semaphore(semaphore)
                        }
                        for semaphore in self.image_available_semaphores.drain(..) {
                                self.device.destroy_semaphore(semaphore)
                        }
                        for framebuffer in self.framebuffers.drain(..) {
                                self.device.destroy_framebuffer(framebuffer);
                        }
                        for image_view in self.image_views.drain(..) {
                                self.device.destroy_image_view(image_view);
                        }
                        // LAST RESORT STYLE CODE, NOT TO BE IMITATED LIGHTLY
                        use core::ptr::read;
                        self.device
                                .destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
                        self.device
                                .free_memory(ManuallyDrop::into_inner(read(&self.memory)));
                        self.device
                                .destroy_pipeline_layout(ManuallyDrop::into_inner(read(
                                        &self.pipeline_layout
                                )));
                        self.device
                                .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(
                                        &self.graphics_pipeline
                                )));
                        self.device.destroy_command_pool(
                                ManuallyDrop::into_inner(read(&self.command_pool)).into_raw(),
                        );
                        self.device
                                .destroy_render_pass(ManuallyDrop::into_inner(read(
                                        &self.render_pass
                                )));
                        self.device
                                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));
                        ManuallyDrop::drop(&mut self.device);
                        ManuallyDrop::drop(&mut self._instance);
                }
        }
}

#[derive(Debug)]
pub struct WinitState {
        pub events_loop: EventsLoop,
        pub window: Window,
}
impl WinitState {
        /// Constructs a new `EventsLoop` and `Window` pair.
        ///
        /// The specified title and size are used, other elements are default.
        /// ## Failure
        /// It's possible for the window creation to fail. This is unlikely.
        pub fn new<T: Into<String>>(title: T, size: LogicalSize) -> Result<Self, CreationError> {
                let events_loop = EventsLoop::new();
                let output = WindowBuilder::new()
                        .with_title(title)
                        .with_dimensions(size)
                        .build(&events_loop);
                output.map(|window| Self {
                        events_loop,
                        window,
                })
        }
}
impl Default for WinitState {
        /// Makes an 800x600 window with the `WINDOW_NAME` value as the title.
        /// ## Panics
        /// If a `CreationError` occurs.
        fn default() -> Self {
                Self::new(
                        WINDOW_NAME,
                        LogicalSize {
                                width: 800.0,
                                height: 600.0,
                        },
                )
                .expect("Could not create a window!")
        }
}

#[derive(Debug, Clone, Default)]
pub struct UserInput {
        pub end_requested: bool,
        pub new_frame_size: Option<(f64, f64)>,
        pub new_mouse_position: Option<(f64, f64)>,
}
impl UserInput {
        pub fn poll_events_loop(events_loop: &mut EventsLoop) -> Self {
                let mut output = UserInput::default();
                events_loop.poll_events(|event| match event {
                        Event::WindowEvent {
                                event: WindowEvent::CloseRequested,
                                ..
                        } => output.end_requested = true,
                        Event::WindowEvent {
                                event: WindowEvent::Resized(logical),
                                ..
                        } => {
                                output.new_frame_size = Some((logical.width, logical.height));
                        }
                        Event::WindowEvent {
                                event: WindowEvent::CursorMoved { position, .. },
                                ..
                        } => {
                                output.new_mouse_position = Some((position.x, position.y));
                        }
                        _ => (),
                });
                output
        }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LocalState {
        pub frame_width: f64,
        pub frame_height: f64,
        pub mouse_x: f64,
        pub mouse_y: f64,
}
impl LocalState {
        pub fn update_from_input(&mut self, input: UserInput) {
                if let Some(frame_size) = input.new_frame_size {
                        self.frame_width = frame_size.0;
                        self.frame_height = frame_size.1;
                }
                if let Some(position) = input.new_mouse_position {
                        self.mouse_x = position.0;
                        self.mouse_y = position.1;
                }
        }
}

fn _do_the_render_clear(
        hal_state: &mut HalState,
        local_state: &LocalState,
) -> Result<(), &'static str> {
        let r = (local_state.mouse_x / local_state.frame_width) as f32;
        let g = (local_state.mouse_y / local_state.frame_height) as f32;
        let b = (r + g) * 0.3;
        let a = 1.0;
        hal_state.draw_clear_frame([r, g, b, a])
}

fn do_the_render(hal_state: &mut HalState, local_state: &LocalState) -> Result<(), &'static str> {
        let x = ((local_state.mouse_x / local_state.frame_width) * 2.0) - 1.0;
        let y = ((local_state.mouse_y / local_state.frame_height) * 2.0) - 1.0;
        let triangle = Triangle {
                points: [[-0.5, 0.5], [-0.5, 0.5], [x as f32, y as f32]],
        };
        hal_state.draw_triangle_frame(triangle)
}

fn main() {
        simple_logger::init().unwrap();

        let mut winit_state = WinitState::default();

        let mut hal_state = match HalState::new(&winit_state.window) {
                Ok(state) => state,
                Err(e) => panic!(e),
        };

        let (frame_width, frame_height) = winit_state
                .window
                .get_inner_size()
                .map(|logical| logical.into())
                .unwrap_or((0.0, 0.0));
        let mut local_state = LocalState {
                frame_width,
                frame_height,
                mouse_x: 0.0,
                mouse_y: 0.0,
        };

        loop {
                let inputs = UserInput::poll_events_loop(&mut winit_state.events_loop);
                if inputs.end_requested {
                        break;
                }
                if inputs.new_frame_size.is_some() {
                        debug!("Window changed size, restarting HalState...");
                        drop(hal_state);
                        hal_state = match HalState::new(&winit_state.window) {
                                Ok(state) => state,
                                Err(e) => panic!(e),
                        };
                }
                local_state.update_from_input(inputs);
                if let Err(e) = do_the_render(&mut hal_state, &local_state) {
                        error!("Rendering Error: {:?}", e);
                        debug!("Auto-restarting HalState...");
                        drop(hal_state);
                        hal_state = match HalState::new(&winit_state.window) {
                                Ok(state) => state,
                                Err(e) => panic!(e),
                        };
                }
        }
}
