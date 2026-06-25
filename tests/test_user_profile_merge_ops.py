from domain.profile.models import UserProfile
from services.profile import user_profile_ops


def test_deep_merge_list_dicts_by_name_identity():
    base = {
        "friends": [
            {"name": "lin", "relationship": "roommate", "years": 8},
        ]
    }
    incoming = {
        "friends": [
            {"name": "lin", "relationship": "best friend"},
            {"name": "chen", "relationship": "colleague"},
        ]
    }

    merged = user_profile_ops._deep_merge(base, incoming)

    assert len(merged["friends"]) == 2
    lin = next(item for item in merged["friends"] if item["name"] == "lin")
    assert lin["relationship"] == "best friend"
    assert lin["years"] == 8


def test_build_merged_user_profile_keeps_existing_and_compacts_tags_habits():
    existing = UserProfile.model_validate(
        {
            "demographic": {"extra": {"friends": [{"name": "lin", "years": 8}]}},
            "preferences": {
                "habits": [
                    {"name": "晨跑", "confidence": 0.9},
                    {"name": "低质量习惯", "confidence": 0.2},
                ]
            },
            "tags": [
                {"name": "科技爱好者", "weight": 0.8, "source": "implicit", "sub_tags": [], "evidences": {}},
                {"name": "临时标签", "weight": 0.3, "source": "implicit", "sub_tags": [], "evidences": {}},
            ],
        }
    )

    incoming = UserProfile.model_validate(
        {
            "demographic": {"extra": {"friends": [{"name": "lin", "relationship": "best friend"}]}},
            "preferences": {"habits": [{"name": "阅读", "confidence": 0.7}]},
            "tags": [
                {"name": "科技爱好者", "weight": 0.95, "source": "explicit", "sub_tags": [], "evidences": {}},
                {"name": "一次性事件", "weight": 0.4, "source": "implicit", "sub_tags": [], "evidences": {}},
            ],
        }
    )

    merged = user_profile_ops._build_merged_user_profile(existing, incoming)

    friends = merged.demographic.extra.get("friends", [])
    print(f"friends: {friends}")
    assert len(friends) == 1
    assert friends[0]["name"] == "lin"
    assert friends[0]["years"] == 8
    assert friends[0]["relationship"] == "best friend"

    habit_names = {h.name for h in merged.preferences.habits}
    print(f"habit_names: {habit_names}")
    assert "晨跑" in habit_names
    assert "阅读" in habit_names
    assert "低质量习惯" not in habit_names

    tag_names = {t.name for t in merged.tags}
    print(f"tag_names: {tag_names}")
    assert "科技爱好者" in tag_names
    assert "临时标签" not in tag_names
    assert "一次性事件" not in tag_names


def test_resolve_identity_keys_supports_wildcard_path():
    keys = user_profile_ops._resolve_identity_keys("demographic.extra.friends")
    assert "name" in keys
    assert "type" not in keys


def test_merge_lists_on_demographic_extra_path_avoids_type_only_over_merge():
    base = [{"type": "friend", "name": "lin"}]
    incoming = [{"type": "friend", "name": "chen"}]

    merged = user_profile_ops._merge_lists(base, incoming, path="demographic.extra.friends")
    print(f"merged: {merged}")

    assert len(merged) == 2


