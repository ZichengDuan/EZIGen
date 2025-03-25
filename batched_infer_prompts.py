import numpy as np
import os

def prompts_n_subjects():
    prompts = ["a asian woman", "a black man"]
    # classname, image path, identifier
    lives = [
      ["woman", "example_images/subjects/lifeifei.png", "lifeifei"],
      ["man", "example_images/subjects/lifeifei.png", "man"],
      ["Chow Chow dog", "example_images/subjects/lifeifei.png", "dog2"],
      ["small corgi dog", "example_images/subjects/lifeifei.png", "dog6"],
      ["brown cat", "example_images/subjects/lifeifei.png", "cat1"],
      ["gray cat", "example_images/subjects/lifeifei.png", "cat2"],
      ["fire fox", "example_images/subjects/lifeifei.png", "fox"],
    ]

    small_objects = [
      ["white regular bowl", "example_images/subjects/lifeifei.png", "berry_bowl"],
      ["red backpack", "example_images/subjects/lifeifei.png", "backpack1"],
      ["gary backpack", "example_images/subjects/lifeifei.png", "backpack2"],
      ["pink sunglasses", "example_images/subjects/lifeifei.png", "pink_sunglasses"],
      ["white tall boot", "example_images/subjects/lifeifei.png", "facny_boot"],
      ["yellow mug", "example_images/subjects/lifeifei.png", "castle"],
    ]

    huge_objects = [
      ["franch castle", "example_images/subjects/lifeifei.png", "castle"],
      ["white sedan car", "example_images/subjects/lifeifei.png", "car"],
    ]

    lives_prompts = [
      "A {} happily sitting in a sunny park filled with flowers and trees.",
      "A {} smiling brightly while holding a colorful balloon.",
      "A {} joyfully reading a book under the shade of a large tree.",
      "A {} energetically chasing a frisbee on a green field.",
      "A {} sitting comfortably on a picnic blanket during a sunny day.",
      "A {} wearing a bright wizard hat and enjoying a ice cream cone.",
      "A {} listening to music with headphones while sitting on a bench.",
      "A {} riding a bicycle along a picturesque lakeside pathway.",
      "A {} playing freely at a lively outdoor festival.",
      "A {} striking a fun pose in front of a vibrant mural.",
      "A {} exploring a busy market full of colorful stalls and treats.",
      "A {} splashing in the waves at a beautiful beach on a summer day.",
      "A {} tasting delicious fruit at a local farmer's market.",
      "A {} tending to flowers in a charming backyard garden.",
      "A {} enjoying a serene moment while watching the sunset."
    ]

    small_objects_prompts = [
        "A {} elegantly placed on a beautifully set dining table near candles.",
        "A {} resting on a windowsill, sunlight streaming through the glass.",
        "A {} surrounded by vibrant flowers in a picturesque garden.",
        "A {} sitting peacefully on a beach towel by the ocean waves.",
        "A {} displayed proudly in a cozy living room filled with warmth.",
        "A {} nestled among a collection of colorful books on a shelf.",
        "A {} hanging gracefully from a tree branch with a gentle breeze.",
        "A {} lying next to a steaming cup of coffee on a desk.",
        "A {} glowing softly beneath fairy lights in a cozy corner.",
        "A {} placed amidst fallen leaves in a scenic autumn setting.",
        "A {} peeking out from inside a stylish shopping bag.",
        "A {} arranged neatly on a shelf adjacent to decorative items.",
        "A {} lit up under bright lights in a bustling shopping mall.",
        "A {} surrounded by snowflakes in a picturesque winter landscape.",
        "A {} placed on a wooden porch during a calm evening.",
        "A {} sitting next to a blooming plant in a sunlit room."
    ]

    huge_objects_prompts = [
        "A {} elegantly placed on a beautifully set dining table near candles.",
        "A {} resting on a windowsill, sunlight streaming through the glass.",
        "A {} surrounded by vibrant flowers in a picturesque garden.",
        "A {} sitting peacefully on a beach towel by the ocean waves.",
        "A {} displayed proudly in a cozy living room filled with warmth.",
        "A {} nestled among a collection of colorful books on a shelf.",
        "A {} hanging gracefully from a tree branch with a gentle breeze.",
        "A {} lying next to a steaming cup of coffee on a desk.",
        "A {} glowing softly beneath fairy lights in a cozy corner.",
        "A {} placed amidst fallen leaves in a scenic autumn setting.",
        "A {} peeking out from inside a stylish shopping bag.",
        "A {} arranged neatly on a shelf adjacent to decorative items.",
        "A {} lit up under bright lights in a bustling shopping mall.",
        "A {} surrounded by snowflakes in a picturesque winter landscape.",
        "A {} placed on a wooden porch during a calm evening.",
        "A {} sitting next to a blooming plant in a sunlit room."
    ]

    huge_objects_prompts = [
        "A majestic {} standing tall against a vibrant sunset sky.",
        "A grand {} covered in snow, reflecting in a tranquil lake.",
        "An ancient {} looming over a dark, stormy landscape.",
        "A breathtaking {} surrounded by colorful wildflowers in bloom.",
        "A historic {} nestled in lush greenery by a flowing river.",
        "A towering {} bathed in golden light at dawn or dusk.",
        "A fairy-tale {} on a hill overlooking a charming village.",
        "A rugged {} with adventurous hikers exploring its rocky trails.",
        "A magnificent {} with tall spires set under a starlit sky.",
        "A serene {} surrounded by wispy clouds and bright blue skies.",
        "A sprawling {} by the sea, waves crashing against its stones.",
        "A dramatic {} with waterfalls cascading down into a lush valley.",
        "A beautiful {} illuminated by soft, warm lights at night.",
        "A giant {} rising majestically over a peaceful countryside.",
        "A vibrant {} reflected in a calm pond during sunset."
    ]

    seeds = [19990121, 19980701, 71, 121, 1998, 1999, 2025, 1234, 3154, 43]
    pairs = []

    for small_object in small_objects:
      class_name, img_path, identifier = small_object
      for prompt in small_objects_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), img_path, identifier, seed))
    
    for huge_object in huge_objects:
      class_name, img_path, identifier = huge_object
      for prompt in huge_objects_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), img_path, identifier, seed))

    for live in lives:
      class_name, img_path, identifier = live
      for prompt in lives_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), img_path, identifier, seed))

    breakpoint()
    return pairs

prompts_n_subjects()