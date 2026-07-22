import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="GAN Digit Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        background-color: #FF6B6B;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<p class="main-header">🎨 GAN Digit Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Generate realistic handwritten digits using Deep Convolutional GAN</p>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.tensorflow.org/images/tf_logo_social.png", width=200)
    st.markdown("---")
    st.markdown("## 🎛️ Controls")

    # Number of images to generate
    num_images = st.slider(
        "Number of digits to generate",
        min_value=1,
        max_value=25,
        value=16,
        step=1
    )

    # 🔧 FIXED: Model expects 100 dimensions
    st.info("ℹ️ Model is trained with 100-dimensional noise vector")
    noise_dim = 100  # Fixed value

    # Temperature/Scale
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values = more random/creative outputs"
    )

    # Seed for reproducibility
    use_seed = st.checkbox("Use fixed seed for reproducibility")
    seed_value = st.number_input("Seed value", min_value=0, max_value=9999, value=42) if use_seed else None

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("""
    - **Model**: DCGAN
    - **Dataset**: MNIST
    - **Epochs**: 200
    - **Parameters**: 2.33M
    - **Input**: 100-dim noise
    - **Output**: 28×28 grayscale
    """)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app uses a trained GAN to generate new handwritten digits. 
    The model was trained on 60,000 MNIST images.
    """)


# Load model function with caching
@st.cache_resource
def load_model():
    """Load the trained generator model"""
    try:
        model = tf.keras.models.load_model('generator_model_final.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Make sure 'generator_model_final.keras' is in the same directory")
        return None


# Generate images function
def generate_images(generator, num_images, noise_dim, temperature=1.0, seed=None):
    """Generate images using the trained generator"""
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # 🔧 Ensure correct dimension
    try:
        expected_dim = generator.input_shape[-1]
        if noise_dim != expected_dim:
            st.warning(f"Model expects {expected_dim} dimensions. Using {expected_dim}.")
            noise_dim = expected_dim
    except:
        # If cannot get input shape, use 100 (default for MNIST GAN)
        noise_dim = 100

    # Generate random noise
    noise = tf.random.normal([num_images, noise_dim])

    # Apply temperature scaling
    noise = noise * temperature

    # Generate images
    with tf.device('/CPU:0'):  # Use CPU for compatibility
        generated_images = generator(noise, training=False)

    # Denormalize from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0

    # Clip values to [0, 1]
    generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)

    return generated_images.numpy()


# Display images in grid
def display_images_grid(images, num_images, cols=4):
    """Display generated images in a grid layout"""
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle('Generated Digits', fontsize=16, fontweight='bold')

    # If only one row, make axes 2D
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                axes[i, j].imshow(images[idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Digit {idx + 1}', fontsize=8)
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    return fig


# Display individual image
def display_individual_image(image, title="Generated Digit"):
    """Display a single image with zoom capability"""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    return fig


# Statistics
def calculate_statistics(images):
    """Calculate and return image statistics"""
    stats = {
        'mean': np.mean(images),
        'std': np.std(images),
        'min': np.min(images),
        'max': np.max(images),
        'sparsity': np.mean(images < 0.1)  # Percentage of very dark pixels
    }
    return stats


# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🎨 Generate Digits", use_container_width=True):
        with st.spinner('Generating digits...'):
            # Load model
            generator = load_model()

            if generator is not None:
                # Generate images
                seed = seed_value if use_seed else None
                generated_images = generate_images(
                    generator,
                    num_images,
                    noise_dim,
                    temperature,
                    seed
                )

                # Store in session state
                st.session_state.generated_images = generated_images
                st.session_state.num_images = num_images

                # Success message
                st.success(f"✅ Successfully generated {num_images} digits!")

                # Display metrics
                stats = calculate_statistics(generated_images)

                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                with col_metrics1:
                    st.metric("Average Brightness", f"{stats['mean']:.3f}")
                with col_metrics2:
                    st.metric("Std Deviation", f"{stats['std']:.3f}")
                with col_metrics3:
                    st.metric("Sparsity", f"{stats['sparsity'] * 100:.1f}%")
                with col_metrics4:
                    st.metric("Dynamic Range", f"{stats['max'] - stats['min']:.3f}")

                # Display grid
                st.markdown("---")
                st.subheader("🖼️ Generated Digits")

                cols_display = min(4, num_images)
                fig_grid = display_images_grid(generated_images, num_images, cols_display)
                st.pyplot(fig_grid)

                # Individual display option
                st.markdown("---")
                st.subheader("🔍 View Individual Digits")

                selected_digit = st.selectbox(
                    "Select a digit to view in detail",
                    range(1, num_images + 1)
                )

                if selected_digit:
                    fig_single = display_individual_image(
                        generated_images[selected_digit - 1],
                        f"Digit {selected_digit}"
                    )
                    st.pyplot(fig_single)

                # Download option
                st.markdown("---")
                st.subheader("💾 Download Options")

                # Convert to PIL Image for download
                img = Image.fromarray((generated_images[0, :, :, 0] * 255).astype(np.uint8), mode='L')
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download First Digit as PNG",
                    data=byte_im,
                    file_name=f"generated_digit_{time.strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

                # Display info
                st.info(
                    f"💡 Tip: Use the sidebar controls to adjust generation parameters. Temperature {temperature} {'(with seed)' if use_seed else '(random)'}")

# If no images generated yet, show placeholder
if 'generated_images' not in st.session_state:
    st.markdown("---")
    col_empty1, col_empty2, col_empty3 = st.columns([1, 2, 1])

    with col_empty2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 20px;">
            <h2 style="color: #666;">👈 Ready to Generate!</h2>
            <p style="color: #999; font-size: 1.1rem;">
                Click the "Generate Digits" button to create new handwritten digits using the trained GAN model.
            </p>
            <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">
                Adjust settings in the sidebar to customize your generation.
            </p>
            <div style="margin-top: 1rem;">
                <span style="background-color: #e9ecef; padding: 0.5rem 1rem; border-radius: 10px; margin: 0.2rem;">
                    🎯 100-dim noise
                </span>
                <span style="background-color: #e9ecef; padding: 0.5rem 1rem; border-radius: 10px; margin: 0.2rem;">
                    🎨 DCGAN
                </span>
                <span style="background-color: #e9ecef; padding: 0.5rem 1rem; border-radius: 10px; margin: 0.2rem;">
                    📊 28×28 px
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.8rem;">
        Built with ❤️ using TensorFlow & Streamlit | GAN for MNIST Digit Generation
    </div>
    """, unsafe_allow_html=True)