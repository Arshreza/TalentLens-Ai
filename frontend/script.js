async function analyze() {
  const name = document.getElementById("name").value;
  const resume = document.getElementById("resume").value;
  const jd = document.getElementById("jd").value;

  if (!resume || !jd) {
    alert("Please fill resume and job description");
    return;
  }

  try {
    const res = await fetch("http://localhost:8000/api/v1/match", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        candidate_name: name,
        resume_text: resume,
        job_description: jd
      })
    });

    const data = await res.json();
    render(data);

  } catch (err) {
    console.log("Backend not running, using mock...");
    render(mock());
  }
}

function render(data) {
  document.getElementById("result").style.display = "block";

  document.getElementById("score").innerText = data.match_score;
  document.getElementById("insight").innerText = data.ai_insight;

  const skillsDiv = document.getElementById("skills");
  skillsDiv.innerHTML = "";

  data.matched_skills.forEach(skill => {
    const el = document.createElement("div");
    el.className = "skill";
    el.innerText = skill.jd_skill;
    skillsDiv.appendChild(el);
  });
}

function mock() {
  return {
    match_score: 82,
    ai_insight: "Strong candidate, missing Kubernetes & LLM experience",
    matched_skills: [
      { jd_skill: "Python" },
      { jd_skill: "TensorFlow" },
      { jd_skill: "Docker" }
    ]
  };
}