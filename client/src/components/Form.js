// dependencies
import React from 'react'

const Form = (props) => {
    // destructure props
    const {
        title,
        setTitle,
        firstName,
        setFirstName,
        lastName,
        setLastName,
        age,
        setAge,
        gender,
        setGender,
        passengerClass,
        setPassengerClass,
        fare,
        setFare,
        cabinLocation,
        setCabinLocation,
        embarkation,
        setEmbarkation,
        withChildren,
        setWithChildren,
        withParents,
        setWithParents,
        withSpouse,
        setWithSpouse,
        withSiblings,
        setWithSiblings,
        transmitting,
        makePrediction
    } = props

    const submit = e => {
        e.preventDefault()

        makePrediction()
    }

    return (
        <form className="row" id="form">
            <div className="col s12 l6">
                <div className="row">
                    <div className="col s12">
                        <h3>Passenger profile</h3>
                    </div>
                </div>
                <div className="row">
                    <div className="input-field col s4">
                        <span className="label">Title</span>
                        <select
                            className="browser-default"
                            name="title"
                            onChange={e => setTitle(e.target.value)}
                            value={title}
                        >
                            <option value="title_mr">Mr</option>
                            <option value="title_mrs">Mrs</option>
                            <option value="title_master">Master</option>
                            <option value="title_miss">Miss</option>
                            <option value="title_other">Other</option>
                        </select>
                        <span className="error-message"></span>
                    </div>
                    <div className="input-field col s8">
                        <span className="label">First Name</span>
                        <input
                            name="firstName"
                            onChange={e => setFirstName(e.target.value)}
                            type="text"
                            value={firstName}
                        />
                        <span className="error-message"></span>
                    </div>
                </div>
                <div className="row">
                    <div className="input-field col s12">
                        <span className="label">Last Name</span>
                        <input
                            name="lastName"
                            onChange={e => setLastName(e.target.value)}
                            type="text"
                            value={lastName}
                        />
                        <span className="error-message"></span>
                    </div>
                </div>
                <div className="row">
                    <p className="range-field col s4">
                        <span className="label">Age</span>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            name="age"
                            onChange={e => setAge(e.target.value)}
                            value={age}
                        />
                        <span className="error-message"></span>
                    </p>
                    <p className="col s2">
                        <span className="number">{age}</span>
                    </p>
                    <div className="input-field col s6">
                        <span className="label">Gender</span>
                        <select
                            className="browser-default"
                            name="gender"
                            onChange={e => setGender(e.target.value)}
                            value={gender}
                        >
                            <option value="gender_male">Male</option>
                            <option value="gender_female">Female</option>
                        </select>
                        <span className="error-message"></span>
                    </div>
                </div>
            </div>
            <div className="col s12 l6">
                <div className="row">
                    <div className="col s12">
                        <h3>Travel options</h3>
                    </div>
                </div>
                <div className="row">
                    <div className="input-field col s6">
                        <span className="label">Passenger Class</span>
                        <select
                            className="browser-default"
                            name="passengerClass"
                            onChange={e => setPassengerClass(e.target.value)}
                            value={passengerClass}
                        >
                            <option value="passenger_class_1">First Class</option>
                            <option value="passenger_class_2">Second Class</option>
                            <option value="passenger_class_3">Third Class</option>
                        </select>
                        <span className="error-message"></span>
                    </div>
                    <p className="range-field col s4">
                        <span className="label">Ticket Price</span>
                        <input
                            type="range"
                            min="0"
                            max="512"
                            name="fare"
                            onChange={e => setFare(e.target.value)}
                            value={fare}
                        />
                        <span className="error-message"></span>
                    </p>
                    <p className="col s2">
                        <span className="number">{fare}</span>
                    </p>
                </div>
                <div className="row">
                    <div className="input-field col s6">
                        <span className="label">Cabin Location</span>
                        <select
                            className="browser-default"
                            name="cabinLocation"
                            onChange={e => setCabinLocation(e.target.value)}
                            value={cabinLocation}
                        >
                            <option value="deck_a">A Deck</option>
                            <option value="deck_b">B Deck</option>
                            <option value="deck_c">C Deck</option>
                            <option value="deck_d">D Deck</option>
                            <option value="deck_e">E Deck</option>
                            <option value="deck_f">F Deck</option>
                            <option value="deck_g">G Deck</option>
                        </select>
                        <span className="error-message"></span>
                    </div>
                    <div className="input-field col s6">
                        <span className="label">Embarkation Point</span>
                        <select
                            className="browser-default"
                            name="embarkation"
                            onChange={e => setEmbarkation(e.target.value)}
                            value={embarkation}
                        >
                            <option value="embarked_s">Southampton</option>
                            <option value="embarked_c">Cherbourg</option>
                            <option value="embarked_q">Queenstown</option>
                        </select>
                        <span className="error-message"></span>
                    </div>
                </div>
                <div className="row">
                    <div className="col s6 center-align switch">
                        <span className="label">Traveling with children?</span>
                        <label>
                            No
                            <input
                                name="withChildren"
                                onChange={e => setWithChildren(e.target.value)}
                                type="checkbox"
                                value={withChildren}
                            />
                            <span className="lever"></span>
                            Yes
                        </label>
                    </div>
                    <div className="col s6 center-align switch">
                        <span className="label">Traveling with parents?</span>
                        <label>
                            No
                            <input
                                name="withParents"
                                onChange={e => setWithParents(e.target.value)}
                                type="checkbox"
                                value={withParents}
                            />
                            <span className="lever"></span>
                            Yes
                        </label>
                    </div>
                </div>
                <div className="row">
                    <div className="col s6 center-align switch">
                        <span className="label">Traveling with a spouse?</span>
                        <label>
                            No
                            <input
                                name="withSpouse"
                                onChange={e => setWithSpouse(e.target.value)}
                                type="checkbox"
                                value={withSpouse}
                            />
                            <span className="lever"></span>
                            Yes
                        </label>
                    </div>
                    <div className="col s6 center-align switch">
                        <span className="label">Traveling with siblings?</span>
                        <label>
                            No
                            <input
                                name="withSiblings"
                                onChange={e => setWithSiblings(e.target.value)}
                                type="checkbox"
                                value={withSiblings}
                            />
                            <span className="lever"></span>
                            Yes
                        </label>
                    </div>
                </div>
            </div>
            <div className="col s12">
                <button
                    className="btn"
                    onClick={submit}
                    type="submit"
                >
                    Submit
                </button>
            </div>
        </form>
    )
}

export default Form
