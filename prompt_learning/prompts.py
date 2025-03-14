
def get_pathological_tissue_level_prompts(multi_templates=True):
    common_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a photo of the hard to see {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a photo of one {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]

    pathology_templates = [
        'a histopathological image of {}.',
        'a microscopic image of {} in tissue.',
        'a pathology slide showing {}.',
        'a high magnification image of {}.',
        'an immunohistochemical staining of {}.',
        'a pathology image of {} with inflammatory cells.',
        'a low magnification image of {}.',
        'a pathology image of {} with cellular atypia.',
        'a pathology image of {} with necrosis.',
        'an H&E stained image of {}.',
        'a pathology image of {} with fibrosis.',
        'a pathology image of {} with neoplastic cells.',
        'a pathology image of {} with metastasis.',
        'a pathology image of {} with infiltrating cells.',
        'a pathology image of {} with granulation tissue.',
        'an image of {} on a pathology slide.',
        'a pathology image of {} with edema.',
        'a pathology image of {} with hemorrhage.',
        'a pathology image of {} with degenerative changes.',
        'a pathology image of {} with angiogenesis.',
    ]

    knowledge_from_chatGPT = {
        "Squamous epithelium": "Flat, plate-like cells with a centrally located nucleus.",
        "Columnar epithelium": "Elongated cells with a basally located, oval-shaped nucleus.",
        "Glandular epithelium": "Cells organized in gland-like structures, secreting various substances.",
        "Adipose tissue": "Large, round cells with a thin rim of cytoplasm and a peripheral nucleus, filled with a lipid droplet.",
        "Fibrous connective tissue": "Dense arrangement of collagen fibers and fibroblast cells with elongated nuclei.",
        "Cartilage": "Chondrocytes embedded in a matrix with a basophilic appearance, arranged in lacunae.",
        "Bone tissue": "Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi.",
        "Skeletal muscle": "Long, cylindrical, multinucleated cells with visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branching, striated cells with a single, centrally located nucleus and intercalated discs between cells.",
        "Neurons": "Large, star-shaped cells with a prominent, round nucleus and several processes extending from the cell body.",
        "Glial cells": "Smaller, supportive cells with a less-defined shape and a small, dark nucleus.",
        "Lymphocytes": "Small, round cells with a large, dark nucleus and a thin rim of cytoplasm.",
        "Germinal centers": "Areas of active lymphocyte proliferation and differentiation, appearing as lighter-stained regions in lymphoid tissue.",
        "Erythrocytes": "Anucleate, biconcave, disc-shaped cells.",
        "Leukocytes": "Nucleated white blood cells with various morphologies, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Large, polygonal cells with a round, centrally located nucleus and abundant cytoplasm.",
        "Sinusoids": "Vascular channels between hepatocytes, lined by endothelial cells and Kupffer cells in liver tissue.",
        "Glomeruli": "Compact, round structures composed of capillaries and specialized cells with a visible Bowman's space in kidney tissue.",
        "Tubules": "Epithelial-lined structures with various cell types, including proximal and distal tubule cells in kidney tissue.",

        "Carcinoma": "Disorganized tissue architecture, cellular atypia, and possible invasion into surrounding tissues in epithelial-derived tissues.",
        "Sarcoma": "Pleomorphic cells, high cellularity, and possible invasion into surrounding tissues in mesenchymal-derived tissues.",
        "Lymphoma": "Atypical lymphocytes, disrupted lymphoid architecture, and possible effacement of normal lymphoid structures.",
        "Leukemia": "Increased number of abnormal white blood cells in blood smears or bone marrow aspirates, with variable size and nuclear morphology.",
        "Glioma": "Atypical glial cells, increased cellularity, possible necrosis, and disruption of normal central nervous system tissue architecture.",
        "Melanoma": "Atypical melanocytes with variable size, shape, and pigmentation, cellular atypia, and invasion of surrounding tissues."
    }

    knowledge_from_chatGPT_natural = {
        "Squamous epithelium": "Thin, flat cells resembling plates, with a nucleus located in the center.",
        "Columnar epithelium": "Tall cells with an oval-shaped nucleus located toward the base.",
        "Glandular epithelium": "Cells arranged in gland-like structures, responsible for secreting various substances.",
        "Adipose tissue": "Round cells with a thin layer of cytoplasm surrounding a large lipid droplet, and a nucleus pushed to the side.",
        "Fibrous connective tissue": "Tightly packed collagen fibers with elongated nuclei in fibroblast cells.",
        "Cartilage": "Chondrocytes found within a basophilic matrix, situated in small spaces called lacunae.",
        "Bone tissue": "Hard, calcified matrix containing osteocytes in lacunae, which are connected by tiny channels called canaliculi.",
        "Skeletal muscle": "Long, tube-shaped cells with multiple nuclei and visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branched, striated cells with a single central nucleus and intercalated discs connecting the cells.",
        "Neurons": "Star-shaped cells with a large, round nucleus and various extensions coming from the cell body.",
        "Glial cells": "Smaller supporting cells with an undefined shape and a small, dark nucleus.",
        "Lymphocytes": "Tiny, round cells with a large, dark nucleus and a thin layer of cytoplasm.",
        "Erythrocytes": "Disc-shaped cells without a nucleus, featuring a biconcave shape.",
        "Leukocytes": "White blood cells with nuclei, displaying a range of shapes, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Sizeable, polygonal cells with a centrally positioned round nucleus and plenty of cytoplasm.",
        "Glomeruli": "Dense, round formations made up of capillaries and special cells, with a visible Bowman's space in kidney tissue.",
        "Tubules": "Structures lined with epithelial cells, containing various cell types like proximal and distal tubule cells in kidney tissue.",
        "Carcinoma": "Cancerous growth originating from epithelial cells, often exhibiting abnormal cell appearance and disordered tissue structure.",
        "Sarcoma": "Cancerous growth arising from mesenchymal cells, such as those found in bone, cartilage, fat, muscle, or blood vessels.",
        "Lymphoma": "Cancerous growth originating from lymphocytes or lymphoid tissue, often marked by unusual lymphocytes and disrupted lymphoid structure.",
        "Leukemia": "Cancerous growth of blood-forming tissues, characterized by a high number of abnormal white blood cells in the blood and bone marrow.",
        "Glioma": "Cancerous growth arising from glial cells in the central nervous system, often displaying abnormal cell appearance, increased cellularity, and tissue decay.",
        "Melanoma": "Cancerous growth originating from melanocytes, often marked by irregular melanocytes, abnormal cell appearance, and invasion into nearby tissues."
    }

    pathology_templates_t = 'an H&E stained image of {}.'
    common_templates_t = 'a photo of the {}.'

    if multi_templates:
        prompts_common_templates = [[common_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", which is {}".format(tissue_description))
             for tissue_type, tissue_description in knowledge_from_chatGPT.items()]
            for pathology_templates_i in pathology_templates]

    else:
        prompts_common_templates = [common_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template = [pathology_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template_withDescription = [pathology_templates_t.format(tissue_type).replace(".", ", which is {}".format(tissue_description)) for tissue_type, tissue_description in knowledge_from_chatGPT.items()]

    return prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription


def get_patch_level_prompts_forCAMELYON(tissue_type='multi'):
    common_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a photo of the hard to see {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a photo of one {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]

    pathology_templates = [
        'a histopathological image of {}.',
        'a microscopic image of {} in tissue.',
        'a pathology slide showing {}.',
        'a high magnification image of {}.',
        'an immunohistochemical staining of {}.',
        'a pathology image of {} with inflammatory cells.',
        'a low magnification image of {}.',
        'a pathology image of {} with cellular atypia.',
        'a pathology image of {} with necrosis.',
        'an H&E stained image of {}.',
        'a pathology image of {} with fibrosis.',
        'a pathology image of {} with neoplastic cells.',
        'a pathology image of {} with metastasis.',
        'a pathology image of {} with infiltrating cells.',
        'a pathology image of {} with granulation tissue.',
        'an image of {} on a pathology slide.',
        'a pathology image of {} with edema.',
        'a pathology image of {} with hemorrhage.',
        'a pathology image of {} with degenerative changes.',
        'a pathology image of {} with angiogenesis.',
    ]

    CAMELYON_tissue_types = {
        "lymphoid tissue": "Lymphoid tissue has a spongy appearance and is found in lymph nodes. It has areas with different shades, representing the cortex and medulla. The cortex is darker and more packed with cells, while the medulla has a lighter, looser arrangement.",
        "metastatic breast cancer cells": "These cancer cells in lymph nodes can appear as single cells, small groups, or even larger clusters. They typically have large, irregular nuclei and a high nucleus-to-cytoplasm ratio, making them stand out from the surrounding tissue.",
        "germinal centers": "Located within the lymph node cortex, germinal centers appear lighter in color compared to the surrounding area. They contain a mix of large and small lymphocytes, giving them a somewhat grainy appearance.",
        "sinus histiocytosis": "This condition shows up as large, irregular-shaped cells with a lot of cytoplasm. These cells, called histiocytes or macrophages, gather in the sinuses of lymph nodes and can sometimes be seen as clumps.",
        "blood vessels": "Blood vessels in lymph nodes have thin, tube-like structures with a lining of flat cells. Depending on their size, they can have slightly different appearances, but they all look like channels running through the tissue.",
        "connective tissue": "Connective tissue can be seen as a network of fibers and cells supporting the lymph nodes. It's found in the capsule and trabeculae, which are structures that help maintain the shape and organization of the lymph node.",
        "fat tissue": "Fat tissue, or adipose tissue, shows up as large, round cells with clear, empty-looking cytoplasm. These cells store energy and can often be found around lymph nodes."
    }

    CAMELYON_tissue_types_simple = {
        "normal": "normal image patch has regularly shaped cells and smaller, lighter nuclei",
        "tumor": "tumor image patch has irregular cancerous cells and larger, darker nuclei"
    }

    pathology_templates_t = 'an H&E stained image of {}.'
    common_templates_t = 'a photo of the {}.'

    if tissue_type == 'multi':
        prompts_common_templates = [[common_templates_i.format(condition) for condition in CAMELYON_tissue_types.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in CAMELYON_tissue_types.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", {}".format(tissue_description))
             for tissue_type, tissue_description in CAMELYON_tissue_types.items()]
            for pathology_templates_i in pathology_templates]
    elif tissue_type == 'simple':
        prompts_common_templates = [[common_templates_i.format(condition) for condition in CAMELYON_tissue_types_simple.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in CAMELYON_tissue_types_simple.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", {}".format(tissue_description))
             for tissue_type, tissue_description in CAMELYON_tissue_types_simple.items()]
            for pathology_templates_i in pathology_templates]
    else:
        print("unknown tissue type: {}".format(tissue_type))
        raise

    return prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription
